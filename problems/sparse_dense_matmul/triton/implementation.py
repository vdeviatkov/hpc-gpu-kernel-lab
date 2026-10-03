"""Triton backend for 28 · Sparse Matrix-Dense Matrix Multiplication. Not implemented yet.

Implement spmm_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def spmm_kernel(
    row_ptr_ptr, col_idx_ptr, values_ptr, b_ptr, out_ptr, m, k, n, nnz, BLOCK: tl.constexpr
):
    pass  # TODO: implement.


def spmm(row_ptr, col_idx, values, b, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError("spmm: Triton backend not implemented (triton/implementation.py)")
    out = torch.empty((row_ptr.numel() - 1, b.shape[1]), dtype=b.dtype, device=b.device)
    m = row_ptr.numel() - 1
    k = b.shape[0]
    n = b.shape[1]
    nnz = values.numel()
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(row_ptr.device):
        spmm_kernel[grid](
            row_ptr, col_idx, values, b, out, m, k, n, nnz, BLOCK=block_size, num_warps=num_warps
        )
    return out
