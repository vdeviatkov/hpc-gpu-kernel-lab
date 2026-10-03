"""Triton backend for 22 · Sparse Matrix-Vector Multiplication. Not implemented yet.

Implement spmv_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def spmv_kernel(
    row_ptr_ptr, col_idx_ptr, values_ptr, x_ptr, out_ptr, m, k, nnz, BLOCK: tl.constexpr
):
    pass  # TODO: implement.


def spmv(row_ptr, col_idx, values, x, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError("spmv: Triton backend not implemented (triton/implementation.py)")
    out = torch.empty(row_ptr.numel() - 1, dtype=values.dtype, device=values.device)
    m = row_ptr.numel() - 1
    k = x.numel()
    nnz = values.numel()
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(row_ptr.device):
        spmv_kernel[grid](
            row_ptr, col_idx, values, x, out, m, k, nnz, BLOCK=block_size, num_warps=num_warps
        )
    return out
