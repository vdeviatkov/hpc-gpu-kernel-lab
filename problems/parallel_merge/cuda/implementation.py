"""CUDA backend for 20 · Parallel Merge. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def merge(a, b):
    if not IMPLEMENTED:
        raise NotImplementedError("merge: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty(a.numel() + b.numel(), dtype=a.dtype, device=a.device)
    load_extension("gpu_lab_parallel_merge", Path(__file__).parent).merge(a, b, out)
    return out
