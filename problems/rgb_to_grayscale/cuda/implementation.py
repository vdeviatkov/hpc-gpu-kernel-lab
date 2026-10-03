"""CUDA backend for 05 · RGB to Grayscale. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def rgb_to_grayscale(image):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "rgb_to_grayscale: CUDA backend not implemented (cuda/kernels.cu)"
        )
    out = torch.empty(image.shape[:2], dtype=image.dtype, device=image.device)
    load_extension("gpu_lab_rgb_to_grayscale", Path(__file__).parent).rgb_to_grayscale(image, out)
    return out
