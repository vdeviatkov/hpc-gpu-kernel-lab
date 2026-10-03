"""Triton backend for 07 · Matrix Copy. Not implemented yet.

Implement matrix_copy_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def matrix_copy_kernel(a_ptr, out_ptr, m, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def matrix_copy(a, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "matrix_copy: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(a)
    m = a.shape[0]
    n = a.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(a.device):
        matrix_copy_kernel[grid](a, out, m, n, BLOCK=block_size, num_warps=num_warps)
    return out
