"""Triton backend for 18 · Histogramming. Not implemented yet.

Implement histogram_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def histogram_kernel(x_ptr, bins, out_ptr, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def histogram(x, bins, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "histogram: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty(bins, dtype=torch.int32, device=x.device)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        histogram_kernel[grid](x, bins, out, n, BLOCK=block_size, num_warps=num_warps)
    return out
