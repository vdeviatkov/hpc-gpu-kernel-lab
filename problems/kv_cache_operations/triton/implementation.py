"""Triton backend for 47 · KV-Cache Update and Paged Access. Not implemented yet.

Implement kv_cache_append_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def kv_cache_append_kernel(
    k_new_ptr,
    v_new_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    positions_ptr,
    b,
    h,
    t,
    d,
    page_size,
    BLOCK: tl.constexpr,
):
    pass  # TODO: implement.


def kv_cache_append(
    k_new, v_new, k_cache, v_cache, block_table, positions, *, block_size=1024, num_warps=4
):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "kv_cache_append: Triton backend not implemented (triton/implementation.py)"
        )
    b = k_new.shape[0]
    h = k_new.shape[1]
    t = k_new.shape[2]
    d = k_new.shape[3]
    page_size = k_cache.shape[2]
    grid = (triton.cdiv(b, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(k_new.device):
        kv_cache_append_kernel[grid](
            k_new,
            v_new,
            k_cache,
            v_cache,
            block_table,
            positions,
            b,
            h,
            t,
            d,
            page_size,
            BLOCK=block_size,
            num_warps=num_warps,
        )
    return (k_cache, v_cache)
