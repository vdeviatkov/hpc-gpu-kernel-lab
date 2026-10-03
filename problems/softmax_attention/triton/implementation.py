"""Triton backend for 40 · Softmax Attention. Not implemented yet.

Implement attention_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr, m, n, d, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def attention(q, k, v, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "attention: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(q)
    m = q.shape[0]
    n = k.shape[0]
    d = q.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(q.device):
        attention_kernel[grid](q, k, v, out, m, n, d, BLOCK=block_size, num_warps=num_warps)
    return out
