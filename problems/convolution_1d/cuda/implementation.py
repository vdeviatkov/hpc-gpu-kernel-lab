"""CUDA conv1d variants, built lazily and launched on the current PyTorch stream.

variant 0 (cuda_naive): x and w from global memory. variant 1 (cuda_const): w staged as
FP32 in __constant__ memory, at most MAX_CONST_TAPS taps. See cuda/kernels.cu.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = True
MAX_CONST_TAPS = 16384  # 64 KiB of FP32; must match kMaxConstTaps in kernels.cu


def extension():
    return load_extension("gpu_lab_convolution_1d", Path(__file__).parent)


def conv1d(x, w, *, variant=0):
    if variant not in (0, 1):
        raise ValueError("variant must be 0 (naive) or 1 (const)")
    if variant == 1 and w.numel() > MAX_CONST_TAPS:
        raise ValueError(f"cuda_const holds at most {MAX_CONST_TAPS} filter taps")
    out = torch.empty(x.numel() - w.numel() + 1, dtype=x.dtype, device=x.device)
    extension().conv1d(x, w, out, variant)
    return out
