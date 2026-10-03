"""Triton backend for 46 · Quantize / Dequantize Pipeline. Not implemented yet.

Implement quantize_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def quantize_kernel(x_ptr, group_size, q_ptr, scale_ptr, m, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def quantize(x, group_size, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "quantize: Triton backend not implemented (triton/implementation.py)"
        )
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty(
        (x.shape[0], x.shape[1] // group_size), dtype=torch.float32, device=x.device
    )
    m = x.shape[0]
    n = x.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        quantize_kernel[grid](x, group_size, q, scale, m, n, BLOCK=block_size, num_warps=num_warps)
    return (q, scale)
