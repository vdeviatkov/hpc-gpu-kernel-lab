"""CUDA backend for 16 · Dot Product. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def dot(a, b):
    if not IMPLEMENTED:
        raise NotImplementedError("dot: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((), dtype=torch.float32, device=a.device)
    load_extension("gpu_lab_dot_product", Path(__file__).parent).dot(a, b, out)
    return out
