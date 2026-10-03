"""Triton backend for 33 · RMS Normalization. Not implemented yet.

Implement rms_norm_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, eps, out_ptr, m, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def rms_norm(x, weight, eps, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "rms_norm: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(x)
    m = x.shape[0]
    n = x.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        rms_norm_kernel[grid](x, weight, eps, out, m, n, BLOCK=block_size, num_warps=num_warps)
    return out
