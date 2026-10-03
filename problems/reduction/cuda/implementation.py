"""CUDA backend for 13 · Reduction. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def reduce_sum(x):
    if not IMPLEMENTED:
        raise NotImplementedError("reduce_sum: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((), dtype=torch.float32, device=x.device)
    load_extension("gpu_lab_reduction", Path(__file__).parent).reduce_sum(x, out)
    return out
