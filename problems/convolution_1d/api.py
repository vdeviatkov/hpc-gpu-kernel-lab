"""09 · 1D Convolution: public entry point.

Lab contract (the LeetGPU statement linked in README.md is not yet verified; its page
renders client-side):

Valid 1-D cross-correlation, no filter flip:

    out[i] = sum_{j < K} x[i + j] * w[j],   0 <= i < N - K + 1,

with products accumulated in FP32 and rounded once to the input dtype. This is what deep
learning frameworks call convolution (F.conv1d).

Inputs are contiguous strided 1-D tensors x (N) and w (K), 1 <= K <= N, of one dtype
(fp32, fp16 or bf16) on one device. Storage offsets are allowed. The result is a fresh
contiguous tensor of N - K + 1 elements on the input device; inputs are not mutated.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "conv1d"
VARIANTS = {
    # One thread per output; x and w read from global memory.
    "cuda_naive": ("cuda", {"variant": 0}),
    # The same, with w staged as FP32 in __constant__ memory (K <= 16384).
    "cuda_const": ("cuda", {"variant": 1}),
    # A shared tile of BLOCK + K - 1 x values per block behind a barrier; w as in naive.
    "cuda_smem": ("cuda", {"variant": 2}),
}
# `cuda` is cuda_naive: the CUDA kernel that accepts every K.
BACKENDS = ("pytorch", *VARIANTS, "cuda", "triton", "jax")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference, per input dtype. Both sides accumulate in
# FP32 and round once, so they differ by at most about one output ulp (covered by rtol: 1
# ulp is <= 2^-10 relative in FP16 and 2^-7 in BF16) plus FP32 summation-order noise (atol).
# The noise grows roughly as sqrt(K); tests/test_convolution_1d.py checks these hold up to
# K = 2047. See docs/design.md, "Tolerances".
TOLERANCES = {
    torch.float32: (0.0001, 0.0001),
    torch.float16: (0.01, 0.01),
    torch.bfloat16: (0.05, 0.05),
}


def validate(x, w):
    for t in (x, w):
        if not isinstance(t, torch.Tensor):
            raise TypeError("expected torch.Tensor")
        if t.layout != torch.strided or t.ndim != 1 or not t.is_contiguous():
            raise ValueError("expected contiguous one-dimensional tensors")
        if t.dtype not in DTYPES.values():
            raise ValueError("supported dtypes: fp32, fp16, bf16")
        if t.requires_grad:
            raise ValueError("this experiment is forward-only; detach inputs first")
        if t.is_neg() or t.is_conj():
            raise ValueError("unresolved negative/conjugate views are unsupported")
    if x.dtype != w.dtype or x.device != w.device:
        raise ValueError("x and w must have the same dtype and device")
    if not 1 <= w.numel() <= x.numel():
        raise ValueError("expected 1 <= K <= N")


def conv1d(x, w, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    validate(x, w)
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, w, **options)
