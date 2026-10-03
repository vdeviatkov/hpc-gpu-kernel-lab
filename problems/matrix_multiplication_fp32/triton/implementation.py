"""Triton backend for 23 · Matrix Multiplication. Not implemented yet.

Implement matmul_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def matmul_kernel(a_ptr, b_ptr, out_ptr, m, k, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def matmul(a, b, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "matmul: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(a.device):
        matmul_kernel[grid](a, b, out, m, k, n, BLOCK=block_size, num_warps=num_warps)
    return out
