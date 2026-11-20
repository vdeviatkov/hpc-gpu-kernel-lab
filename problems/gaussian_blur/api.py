"""10 · Gaussian Blur: public entry point.

Contract, verified against the LeetGPU statement and checker (README.md, source; details in
docs/design.md §0) and widened to FP16/BF16 for the lab:

Same-size 2-D cross-correlation of an (H, W) image with a (KH, KW) kernel, KH and KW odd,
zero padding outside the image, no kernel flip:

    out[i, j] = sum_{m < KH} sum_{n < KW} image[i + m - KH//2, j + n - KW//2] * kernel[m, n]

where image values outside [0, H) x [0, W) are 0. Products are accumulated in FP32 and
rounded once to the input dtype. This is F.conv2d with padding (KH//2, KW//2).

Inputs are contiguous strided 2-D tensors of one dtype (fp32, fp16 or bf16) on one device,
H, W >= 1 and KH, KW >= 1 odd. The kernel may be larger than the image. Storage offsets
are allowed. The result is a fresh contiguous (H, W) tensor on the input device; inputs are
not mutated.

The kernel is any 2-D array: the source calls it Gaussian, but its performance case is a
uniform random, non-separable kernel. A backend may run two 1-D passes only when
`separable_factors` accepts the kernel, and it must keep the intermediate in FP32, so the
output is still rounded once.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "gaussian_blur"
VARIANTS = {
    # One output pixel per thread, the full KH x KW loop, image and kernel from global memory.
    "cuda_direct": ("cuda", {"variant": 0}),
}
# `cuda` is cuda_direct until a faster kernel valid for every kernel exists.
BACKENDS = ("pytorch", *VARIANTS, "cuda", "triton", "jax")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference, per input dtype. FP32 matches the source
# checker (1e-5, 1e-5). Both sides accumulate in FP32 and round once, so they differ by
# about one output ulp (rtol: 2^-10 relative in FP16, 2^-7 in BF16) plus FP32 summation-
# order and factorization noise, which grows with the number of terms, KH * KW (atol).
# tests/test_gaussian_blur.py checks these hold up to 31 x 31; see docs/design.md,
# "Tolerances".
TOLERANCES = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (0.01, 0.01),
    torch.bfloat16: (0.05, 0.05),
}

# A kernel is separable for dtype d when its best rank-1 factorization with FP32 factors
# leaves a residual ||kernel - outer(col, row)||_F <= SEPARABLE_RTOL[d] * ||kernel||_F.
# 4 unit roundoffs of d: a separable kernel rounded to d (and its factors rounded to FP32)
# measures <= 0.93 u_d, the source's random kernels >= 0.2. See docs/design.md.
SEPARABLE_RTOL = {dtype: 4 * torch.finfo(dtype).eps / 2 for dtype in DTYPES.values()}


def validate(image, kernel):
    for t in (image, kernel):
        if not isinstance(t, torch.Tensor):
            raise TypeError("expected torch.Tensor")
        if t.layout != torch.strided or t.ndim != 2 or not t.is_contiguous():
            raise ValueError("expected contiguous two-dimensional tensors")
        if t.dtype not in DTYPES.values():
            raise ValueError("supported dtypes: fp32, fp16, bf16")
        if t.requires_grad:
            raise ValueError("this experiment is forward-only; detach inputs first")
        if t.is_neg() or t.is_conj():
            raise ValueError("unresolved negative/conjugate views are unsupported")
    if image.dtype != kernel.dtype or image.device != kernel.device:
        raise ValueError("image and kernel must have the same dtype and device")
    if image.numel() == 0:
        raise ValueError("expected H, W >= 1")
    kh, kw = kernel.shape
    if kh < 1 or kw < 1 or kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("expected odd kernel dimensions KH, KW >= 1")


def separable_factors(kernel):
    """FP32 factors (col (KH,), row (KW,)) with kernel ~= outer(col, row), or None.

    The rank-1 term of the FP64 SVD, sigma_1 u_1 v_1^T, is split evenly as
    (sqrt(sigma_1) u_1) (sqrt(sigma_1) v_1)^T and rounded to FP32. The kernel is accepted
    when the residual against those FP32 factors, computed exactly in FP64, is within
    SEPARABLE_RTOL of its norm. sigma_2 / sigma_1 is at most that relative residual (plus
    the factors' rounding), so the test is the sigma_2 / sigma_1 < eps rule made
    rigorous about everything a two-pass backend discards. Runs on the host: a CUDA
    kernel is copied back, which synchronizes the stream.
    """
    k = kernel.detach().to("cpu", torch.float64)
    norm = torch.linalg.matrix_norm(k).item()
    if norm == 0:
        kh, kw = k.shape
        return torch.zeros(kh), torch.zeros(kw)
    u, s, vh = torch.linalg.svd(k)
    scale = s[0].sqrt()
    col, row = u[:, 0] * scale, vh[0] * scale
    if col.sum() < 0:  # SVD sign is arbitrary; keep blur factors positive
        col, row = -col, -row
    col, row = col.float(), row.float()
    residual = torch.linalg.matrix_norm(k - torch.outer(col.double(), row.double())).item()
    if residual > SEPARABLE_RTOL[kernel.dtype] * norm:
        return None
    return col, row


def gaussian_blur(image, kernel, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    validate(image, kernel)
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, image, kernel, **options)
