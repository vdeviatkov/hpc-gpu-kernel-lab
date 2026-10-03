"""CUDA backend for 04 · Interleave Arrays. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def interleave(a, b):
    if not IMPLEMENTED:
        raise NotImplementedError("interleave: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty(2 * a.numel(), dtype=a.dtype, device=a.device)
    load_extension("gpu_lab_interleave_arrays", Path(__file__).parent).interleave(a, b, out)
    return out
