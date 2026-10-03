"""Triton backend for 42 · Multi-Head Attention. Not implemented yet.

Implement mha_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def mha_kernel(q_ptr, k_ptr, v_ptr, out_ptr, b, h, s, d, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def mha(q, k, v, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError("mha: Triton backend not implemented (triton/implementation.py)")
    out = torch.empty_like(q)
    b = q.shape[0]
    h = q.shape[1]
    s = q.shape[2]
    d = q.shape[3]
    grid = (triton.cdiv(b, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(q.device):
        mha_kernel[grid](q, k, v, out, b, h, s, d, BLOCK=block_size, num_warps=num_warps)
    return out
