"""Triton backend for 17 · Prefix Sum. Not implemented yet.

Implement inclusive_scan_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def inclusive_scan_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def inclusive_scan(x, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "inclusive_scan: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        inclusive_scan_kernel[grid](x, out, n, BLOCK=block_size, num_warps=num_warps)
    return out
