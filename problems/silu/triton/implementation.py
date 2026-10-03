"""Triton backend for 30 · Sigmoid Linear Unit. Not implemented yet.

Implement silu_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def silu(x, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError("silu: Triton backend not implemented (triton/implementation.py)")
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        silu_kernel[grid](x, out, n, BLOCK=block_size, num_warps=num_warps)
    return out
