"""Triton backend for 13 · Reduction. Not implemented yet.

Implement reduce_sum_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def reduce_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def reduce_sum(x, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "reduce_sum: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty((), dtype=torch.float32, device=x.device)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        reduce_sum_kernel[grid](x, out, n, BLOCK=block_size, num_warps=num_warps)
    return out
