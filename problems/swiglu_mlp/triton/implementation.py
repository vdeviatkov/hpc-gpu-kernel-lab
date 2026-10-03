"""Triton backend for 39 · SwiGLU MLP Block. Not implemented yet.

Implement swiglu_mlp_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def swiglu_mlp_kernel(
    x_ptr, w_gate_ptr, w_up_ptr, w_down_ptr, out_ptr, m, d, f, BLOCK: tl.constexpr
):
    pass  # TODO: implement.


def swiglu_mlp(x, w_gate, w_up, w_down, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "swiglu_mlp: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(x)
    m = x.shape[0]
    d = x.shape[1]
    f = w_gate.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        swiglu_mlp_kernel[grid](
            x, w_gate, w_up, w_down, out, m, d, f, BLOCK=block_size, num_warps=num_warps
        )
    return out
