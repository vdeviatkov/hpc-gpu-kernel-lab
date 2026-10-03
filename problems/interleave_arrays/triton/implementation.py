"""Triton backend for 04 · Interleave Arrays. Not implemented yet.

Implement interleave_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def interleave_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def interleave(a, b, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "interleave: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty(2 * a.numel(), dtype=a.dtype, device=a.device)
    n = a.numel()
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(a.device):
        interleave_kernel[grid](a, b, out, n, BLOCK=block_size, num_warps=num_warps)
    return out
