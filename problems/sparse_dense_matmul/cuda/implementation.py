"""CUDA backend for 28 · Sparse Matrix-Dense Matrix Multiplication. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def spmm(row_ptr, col_idx, values, b):
    if not IMPLEMENTED:
        raise NotImplementedError("spmm: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((row_ptr.numel() - 1, b.shape[1]), dtype=b.dtype, device=b.device)
    load_extension("gpu_lab_sparse_dense_matmul", Path(__file__).parent).spmm(
        row_ptr, col_idx, values, b, out
    )
    return out
