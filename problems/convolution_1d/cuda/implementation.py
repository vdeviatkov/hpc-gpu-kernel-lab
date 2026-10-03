"""CUDA backend for 09 · 1D Convolution. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def conv1d(x, w):
    if not IMPLEMENTED:
        raise NotImplementedError("conv1d: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty(x.numel() - w.numel() + 1, dtype=x.dtype, device=x.device)
    load_extension("gpu_lab_convolution_1d", Path(__file__).parent).conv1d(x, w, out)
    return out
