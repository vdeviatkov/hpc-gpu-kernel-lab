"""Triton backend for 45 · Decaying Causal Attention. Not implemented yet.

Implement decaying_attention_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def decaying_attention_kernel(q_ptr, k_ptr, v_ptr, gamma, out_ptr, s, d, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def decaying_attention(q, k, v, gamma, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "decaying_attention: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(q)
    s = q.shape[0]
    d = q.shape[1]
    grid = (triton.cdiv(s, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(q.device):
        decaying_attention_kernel[grid](
            q, k, v, gamma, out, s, d, BLOCK=block_size, num_warps=num_warps
        )
    return out
