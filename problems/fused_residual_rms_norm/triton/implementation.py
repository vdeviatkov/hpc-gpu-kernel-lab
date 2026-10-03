"""Triton backend for 36 · Fused Residual Add and RMS Norm. Not implemented yet.

Implement fused_add_rms_norm_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def fused_add_rms_norm_kernel(
    x_ptr, residual_ptr, weight_ptr, eps, out_ptr, residual_out_ptr, m, n, BLOCK: tl.constexpr
):
    pass  # TODO: implement.


def fused_add_rms_norm(x, residual, weight, eps, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "fused_add_rms_norm: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    m = x.shape[0]
    n = x.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        fused_add_rms_norm_kernel[grid](
            x, residual, weight, eps, out, residual_out, m, n, BLOCK=block_size, num_warps=num_warps
        )
    return (out, residual_out)
