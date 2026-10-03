"""Triton backend for 19 · Stream Compaction. Not implemented yet.

Implement compact_positive_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def compact_positive_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def compact_positive(x, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "compact_positive: Triton backend not implemented (triton/implementation.py)"
        )
    # TODO: Allocate the worst case, count kept elements on device, return out[:count].
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        compact_positive_kernel[grid](x, out, n, BLOCK=block_size, num_warps=num_warps)
    return out
