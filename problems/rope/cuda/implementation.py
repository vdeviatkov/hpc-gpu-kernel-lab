"""CUDA backend for 38 · Rotary Positional Embedding. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def rope(x, cos, sin):
    if not IMPLEMENTED:
        raise NotImplementedError("rope: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(x)
    load_extension("gpu_lab_rope", Path(__file__).parent).rope(x, cos, sin, out)
    return out
