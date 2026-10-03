"""Triton backend for 41 · Causal Self-Attention. Not implemented yet.

Implement causal_attention_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def causal_attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr, s, d, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def causal_attention(q, k, v, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "causal_attention: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(q)
    s = q.shape[0]
    d = q.shape[1]
    grid = (triton.cdiv(s, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(q.device):
        causal_attention_kernel[grid](q, k, v, out, s, d, BLOCK=block_size, num_warps=num_warps)
    return out
