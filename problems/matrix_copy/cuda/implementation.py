"""CUDA backend for 07 · Matrix Copy. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def matrix_copy(a):
    if not IMPLEMENTED:
        raise NotImplementedError("matrix_copy: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(a)
    load_extension("gpu_lab_matrix_copy", Path(__file__).parent).matrix_copy(a, out)
    return out
