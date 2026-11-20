"""CUDA Gaussian blur variants, built lazily and launched on the current PyTorch stream.

variant 0 (cuda_direct): one output pixel per thread, the full KH x KW loop, image and
kernel read from global memory. variant 1 (cuda_direct_const): the same loop with the
kernel converted to FP32 in __constant__ memory, at most MAX_CONST_TAPS taps. See
cuda/kernels.cu.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = True
VARIANT_NAMES = ("direct", "direct_const")
MAX_CONST_TAPS = 16384  # 64 KiB of FP32; must match kMaxConstTaps in kernels.cu


def extension():
    return load_extension("gpu_lab_gaussian_blur", Path(__file__).parent)


def gaussian_blur(image, kernel, *, variant=0):
    if variant not in range(len(VARIANT_NAMES)):
        raise ValueError("variant must be 0 (direct) or 1 (direct_const)")
    if variant == 1 and kernel.numel() > MAX_CONST_TAPS:
        raise ValueError(f"cuda_direct_const holds at most {MAX_CONST_TAPS} kernel taps")
    out = torch.empty_like(image, memory_format=torch.contiguous_format)
    extension().gaussian_blur(image, kernel, out, variant)
    return out
