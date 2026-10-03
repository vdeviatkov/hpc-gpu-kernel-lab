"""Triton backend for 14 · Count Array Element. Not implemented yet.

Implement count_equal_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def count_equal_kernel(x_ptr, value, out_ptr, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def count_equal(x, value, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "count_equal: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty((), dtype=torch.int32, device=x.device)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        count_equal_kernel[grid](x, value, out, n, BLOCK=block_size, num_warps=num_warps)
    return out
