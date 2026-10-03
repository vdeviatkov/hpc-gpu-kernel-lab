"""CUDA backend for 35 · Batch Normalization. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def batch_norm(x, weight, bias, eps):
    if not IMPLEMENTED:
        raise NotImplementedError("batch_norm: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(x)
    load_extension("gpu_lab_batch_norm", Path(__file__).parent).batch_norm(
        x, weight, bias, eps, out
    )
    return out
