"""Triton backend for 31 · Swish-Gated Linear Unit. Not implemented yet.

Implement swiglu_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def swiglu_kernel(x_ptr, out_ptr, m, d, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def swiglu(x, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "swiglu: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)
    m = x.shape[0]
    d = x.shape[1] // 2
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        swiglu_kernel[grid](x, out, m, d, BLOCK=block_size, num_warps=num_warps)
    return out
