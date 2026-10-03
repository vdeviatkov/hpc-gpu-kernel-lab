"""Triton backend for 43 · Attention with Linear Biases. Not implemented yet.

Implement alibi_attention_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def alibi_attention_kernel(
    q_ptr, k_ptr, v_ptr, slopes_ptr, out_ptr, b, h, s, d, BLOCK: tl.constexpr
):
    pass  # TODO: implement.


def alibi_attention(q, k, v, slopes, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "alibi_attention: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(q)
    b = q.shape[0]
    h = q.shape[1]
    s = q.shape[2]
    d = q.shape[3]
    grid = (triton.cdiv(b, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(q.device):
        alibi_attention_kernel[grid](
            q, k, v, slopes, out, b, h, s, d, BLOCK=block_size, num_warps=num_warps
        )
    return out
