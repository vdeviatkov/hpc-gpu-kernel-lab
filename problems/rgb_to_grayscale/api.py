"""05 · RGB to Grayscale: public entry point.

Contract, verified against the LeetGPU statement (challenges/easy/66_rgb_to_grayscale):

    gray[h, w] = 0.299 * R[h, w] + 0.587 * G[h, w] + 0.114 * B[h, w]

- Source: an H x W image of interleaved RGB FP32 values (R, G, B, R, G, B, ...), so
  H * W * 3 values in, H * W values out; 1 <= W, H <= 4096, at most 4,194,304 pixels,
  values in [0, 255]. The performance test is 2048 x 2048 with integer values 0-255; the
  tolerance is 1e-5. The exact coefficients above are required.
- Here the image is a contiguous (H, W, 3) tensor and the result a new (H, W) tensor: the
  same memory layout as the source's flat arrays.
- Lab extensions: FP16/BF16 (computed in FP32, rounded to the image dtype), empty images,
  and images larger than 4,194,304 pixels (for DRAM-bound measurements).
- Implementations may differ in the last bit (fused multiply-add, summation order), so
  results are compared within the tolerances below rather than bit for bit.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "rgb_to_grayscale"
VARIANTS = {
    # Faster PyTorch baselines; "pytorch" itself is the reference (several separate kernels).
    "pytorch_matmul": ("pytorch", {"method": "matmul"}),  # image @ w, one cuBLAS kernel
    "pytorch_compile": ("pytorch", {"method": "compile"}),  # torch.compile, one fused kernel
    "cuda_pixel": ("cuda", {"variant": 0}),  # one thread per pixel, stride-3 loads
    # one thread per 4 (FP32) or 8 (FP16/BF16) pixels: three 16-byte loads, one 16-byte store;
    # a misaligned image runs cuda_pixel.
    "cuda_vec": ("cuda", {"variant": 1}),
    # a block stages each 256-pixel tile through shared memory with 16-byte loads; a
    # misaligned image runs cuda_pixel.
    "cuda_smem": ("cuda", {"variant": 2}),
    # converts to R, G, B planes first (a second kernel), then reads each plane coalesced.
    "cuda_planar": ("cuda", {"variant": 3}),
}
BACKENDS = ("pytorch", *VARIANTS, "triton")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference. FP32 uses the source's 1e-5. For 2-byte types
# a different rounding of the FP32 result can move the output by one unit in the last place:
# at values up to 255 that is 0.125 in FP16 and 1.0 in BF16.
TOLERANCES = {
    torch.float32: (1e-05, 1e-05),
    torch.float16: (2e-3, 0.125),
    torch.bfloat16: (8e-3, 1.0),
}


def rgb_to_grayscale(image, *, backend="pytorch", **options):
    """Return the (H, W) grayscale image of a contiguous (H, W, 3) RGB image."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("expected torch.Tensor")
    if image.layout != torch.strided or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) RGB tensor")
    if not image.is_contiguous():
        raise ValueError("expected a contiguous (interleaved RGB) tensor")
    if image.dtype not in DTYPES.values():
        raise ValueError("supported dtypes: fp32, fp16, bf16")
    if image.requires_grad:
        raise ValueError("this operation is forward-only; detach the input first")
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, image, **options)
