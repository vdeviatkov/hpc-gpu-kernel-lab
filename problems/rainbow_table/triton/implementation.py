"""Unsigned Triton hash with looped and statically unrolled round variants."""

import torch
import triton
import triton.language as tl


@triton.jit
def rainbow_table_kernel(
    x, out, n, rounds, ROUNDS: tl.constexpr, UNROLL: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + i, mask=i < n, other=0).to(tl.uint32)
    if UNROLL:
        for _ in tl.static_range(ROUNDS):
            state = tl.full((BLOCK,), 2166136261, tl.uint32)
            for k in tl.static_range(4):
                state = (state ^ ((value >> (8 * k)) & 255)) * 16777619
            value = state
    else:
        for _ in range(rounds):
            state = tl.full((BLOCK,), 2166136261, tl.uint32)
            for k in tl.static_range(4):
                state = (state ^ ((value >> (8 * k)) & 255)) * 16777619
            value = state
    tl.store(out + i, value, mask=i < n)


def rainbow_table(x, rounds, *, block_size=256, num_warps=4, unroll=False):
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    out = torch.empty(x.shape, dtype=torch.uint32, device=x.device)
    if x.numel():
        with torch.cuda.device(x.device):
            rainbow_table_kernel[(triton.cdiv(x.numel(), block_size),)](
                x,
                out,
                x.numel(),
                rounds,
                ROUNDS=rounds if unroll else 0,
                UNROLL=unroll,
                BLOCK=block_size,
                num_warps=num_warps,
            )
    return out
