"""Triton backend for 35 · Batch Normalization. Not implemented yet.

Implement batch_norm_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def batch_norm_kernel(x_ptr, weight_ptr, bias_ptr, eps, out_ptr, n, c, hw, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def batch_norm(x, weight, bias, eps, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "batch_norm: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(x)
    n = x.shape[0]
    c = x.shape[1]
    hw = x.shape[2] * x.shape[3]
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        batch_norm_kernel[grid](
            x, weight, bias, eps, out, n, c, hw, BLOCK=block_size, num_warps=num_warps
        )
    return out
