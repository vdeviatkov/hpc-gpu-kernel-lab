"""CUDA backend for 21 · Dense GEMV. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def gemv(a, x):
    if not IMPLEMENTED:
        raise NotImplementedError("gemv: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty(a.shape[0], dtype=a.dtype, device=a.device)
    load_extension("gpu_lab_dense_gemv", Path(__file__).parent).gemv(a, x, out)
    return out
