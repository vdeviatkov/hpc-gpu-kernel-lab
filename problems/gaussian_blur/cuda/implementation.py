"""CUDA Gaussian blur variants, built lazily and launched on the current PyTorch stream.

variant 0 (cuda_direct): one output pixel per thread, the full KH x KW loop, image and
kernel read from global memory. See cuda/kernels.cu.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = True
VARIANT_NAMES = ("direct",)


def extension():
    return load_extension("gpu_lab_gaussian_blur", Path(__file__).parent)


def gaussian_blur(image, kernel, *, variant=0):
    if variant not in range(len(VARIANT_NAMES)):
        raise ValueError(f"variant must be 0 ({VARIANT_NAMES[0]})")
    out = torch.empty_like(image, memory_format=torch.contiguous_format)
    extension().gaussian_blur(image, kernel, out, variant)
    return out
