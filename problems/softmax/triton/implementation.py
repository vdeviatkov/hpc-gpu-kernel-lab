"""Triton backend for 32 · Softmax. Not implemented yet.

Implement softmax_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def softmax_kernel(x_ptr, out_ptr, m, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def softmax(x, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "softmax: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(x)
    m = x.shape[0]
    n = x.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        softmax_kernel[grid](x, out, m, n, BLOCK=block_size, num_warps=num_warps)
    return out
