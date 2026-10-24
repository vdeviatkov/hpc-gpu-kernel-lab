"""Tiled interleaving with contiguous loads and one contiguous output stream.

Each program reads BLOCK values from each input, interleaves them in registers,
then writes 2 * BLOCK values. Tile size and warp count are explicit; no autotuning.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def interleave_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    start = tl.program_id(0).to(tl.int64) * BLOCK
    i = start + tl.arange(0, BLOCK)
    a = tl.load(a_ptr + i, mask=i < n, other=0)
    b = tl.load(b_ptr + i, mask=i < n, other=0)
    values = tl.interleave(a, b)
    out_i = 2 * start + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + out_i, values, mask=out_i < 2 * n)


def interleave(a, b, *, block_size=1024, num_warps=4):
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    out = torch.empty(2 * a.numel(), dtype=a.dtype, device=a.device)
    n = a.numel()
    if n:
        with torch.cuda.device(a.device):
            interleave_kernel[(triton.cdiv(n, block_size),)](
                a, b, out, n, BLOCK=block_size, num_warps=num_warps
            )
    return out
