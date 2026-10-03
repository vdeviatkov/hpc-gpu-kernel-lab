"""CUDA backend for 31 · Swish-Gated Linear Unit. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def swiglu(x):
    if not IMPLEMENTED:
        raise NotImplementedError("swiglu: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)
    load_extension("gpu_lab_swiglu_activation", Path(__file__).parent).swiglu(x, out)
    return out
