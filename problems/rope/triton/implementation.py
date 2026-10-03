"""Triton backend for 38 · Rotary Positional Embedding. Not implemented yet.

Implement rope_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, b, s, h, d, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def rope(x, cos, sin, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError("rope: Triton backend not implemented (triton/implementation.py)")
    out = torch.empty_like(x)
    b = x.shape[0]
    s = x.shape[1]
    h = x.shape[2]
    d = x.shape[3]
    grid = (triton.cdiv(b, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        rope_kernel[grid](x, cos, sin, out, b, s, h, d, BLOCK=block_size, num_warps=num_warps)
    return out
