"""Triton backend for 44 · Grouped Query Attention. Not implemented yet.

Implement gqa_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def gqa_kernel(q_ptr, k_ptr, v_ptr, out_ptr, b, hq, hkv, s, d, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def gqa(q, k, v, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError("gqa: Triton backend not implemented (triton/implementation.py)")
    out = torch.empty_like(q)
    b = q.shape[0]
    hq = q.shape[1]
    hkv = k.shape[1]
    s = q.shape[2]
    d = q.shape[3]
    grid = (triton.cdiv(b, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(q.device):
        gqa_kernel[grid](q, k, v, out, b, hq, hkv, s, d, BLOCK=block_size, num_warps=num_warps)
    return out
