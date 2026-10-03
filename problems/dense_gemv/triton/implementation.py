"""Triton backend for 21 · Dense GEMV. Not implemented yet.

Implement gemv_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def gemv_kernel(a_ptr, x_ptr, out_ptr, m, k, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def gemv(a, x, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError("gemv: Triton backend not implemented (triton/implementation.py)")
    out = torch.empty(a.shape[0], dtype=a.dtype, device=a.device)
    m = a.shape[0]
    k = a.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(a.device):
        gemv_kernel[grid](a, x, out, m, k, BLOCK=block_size, num_warps=num_warps)
    return out
