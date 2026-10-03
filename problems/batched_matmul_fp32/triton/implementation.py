"""Triton backend for 24 · Batched Matrix Multiplication — FP32. Not implemented yet.

Implement bmm_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def bmm_kernel(a_ptr, b_ptr, out_ptr, batch, m, k, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def bmm(a, b, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError("bmm: Triton backend not implemented (triton/implementation.py)")
    out = torch.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=a.dtype, device=a.device)
    batch = a.shape[0]
    m = a.shape[1]
    k = a.shape[2]
    n = b.shape[2]
    grid = (triton.cdiv(batch, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(a.device):
        bmm_kernel[grid](a, b, out, batch, m, k, n, BLOCK=block_size, num_warps=num_warps)
    return out
