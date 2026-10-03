"""CUDA backend for 30 · Sigmoid Linear Unit. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def silu(x):
    if not IMPLEMENTED:
        raise NotImplementedError("silu: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(x)
    load_extension("gpu_lab_silu", Path(__file__).parent).silu(x, out)
    return out
