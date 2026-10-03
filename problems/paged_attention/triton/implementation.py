"""Triton backend for 49 · Paged Attention. Not implemented yet.

Implement paged_attention_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def paged_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    context_lens_ptr,
    out_ptr,
    b,
    h,
    d,
    page_size,
    max_blocks,
    BLOCK: tl.constexpr,
):
    pass  # TODO: implement.


def paged_attention(
    q, k_cache, v_cache, block_table, context_lens, *, block_size=1024, num_warps=4
):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "paged_attention: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(q)
    b = q.shape[0]
    h = q.shape[1]
    d = q.shape[2]
    page_size = k_cache.shape[2]
    max_blocks = block_table.shape[1]
    grid = (triton.cdiv(b, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(q.device):
        paged_attention_kernel[grid](
            q,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            out,
            b,
            h,
            d,
            page_size,
            max_blocks,
            BLOCK=block_size,
            num_warps=num_warps,
        )
    return out
