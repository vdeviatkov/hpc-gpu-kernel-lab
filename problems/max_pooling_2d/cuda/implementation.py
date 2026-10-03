"""CUDA backend for 11 · 2D Max Pooling. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def max_pool2d(x, kernel_size, stride, padding):
    if not IMPLEMENTED:
        raise NotImplementedError("max_pool2d: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty(
        (
            *x.shape[:2],
            (x.shape[2] + 2 * padding - kernel_size) // stride + 1,
            (x.shape[3] + 2 * padding - kernel_size) // stride + 1,
        ),
        dtype=x.dtype,
        device=x.device,
    )
    load_extension("gpu_lab_max_pooling_2d", Path(__file__).parent).max_pool2d(
        x, kernel_size, stride, padding, out
    )
    return out
