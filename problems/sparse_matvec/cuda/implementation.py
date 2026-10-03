"""CUDA backend for 22 · Sparse Matrix-Vector Multiplication. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def spmv(row_ptr, col_idx, values, x):
    if not IMPLEMENTED:
        raise NotImplementedError("spmv: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty(row_ptr.numel() - 1, dtype=values.dtype, device=values.device)
    load_extension("gpu_lab_sparse_matvec", Path(__file__).parent).spmv(
        row_ptr, col_idx, values, x, out
    )
    return out
