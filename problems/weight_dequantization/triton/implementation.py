"""Triton backend for 12 · Weight Dequantization. Not implemented yet.

Implement dequantize_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def dequantize_kernel(q_ptr, scale_ptr, quant_block, out_ptr, m, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def dequantize(q, scale, quant_block, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "dequantize: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty(q.shape, dtype=scale.dtype, device=q.device)
    m = q.shape[0]
    n = q.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(q.device):
        dequantize_kernel[grid](
            q, scale, quant_block, out, m, n, BLOCK=block_size, num_warps=num_warps
        )
    return out
