"""Triton backend for 25 · General Matrix Multiplication (GEMM) — FP16. Not implemented yet.

Implement gemm_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, alpha, beta, out_ptr, m, k, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def gemm(a, b, c, alpha, beta, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError("gemm: Triton backend not implemented (triton/implementation.py)")
    out = torch.empty_like(c)
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(a.device):
        gemm_kernel[grid](a, b, c, alpha, beta, out, m, k, n, BLOCK=block_size, num_warps=num_warps)
    return out
