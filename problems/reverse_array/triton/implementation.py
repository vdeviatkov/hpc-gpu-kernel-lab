"""Triton backend for 03 · Reverse Array. Not implemented yet.

Implement reverse__kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def reverse__kernel(x_ptr, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def reverse_(x, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "reverse_: Triton backend not implemented (triton/implementation.py)"
        )
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        reverse__kernel[grid](x, n, BLOCK=block_size, num_warps=num_warps)
    return x
