"""Triton backend for 09 · 1D Convolution. Not implemented yet.

Implement conv1d_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def conv1d_kernel(x_ptr, w_ptr, out_ptr, n, k, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def conv1d(x, w, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "conv1d: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty(x.numel() - w.numel() + 1, dtype=x.dtype, device=x.device)
    n = x.numel()
    k = w.numel()
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        conv1d_kernel[grid](x, w, out, n, k, BLOCK=block_size, num_warps=num_warps)
    return out
