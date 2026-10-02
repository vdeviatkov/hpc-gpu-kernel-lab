"""Explicit tile/warp selection; no autotuner or hidden configuration sweep."""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add(a, b, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + i, i < n, other=0).to(tl.float32)
    y = tl.load(b + i, i < n, other=0).to(tl.float32)
    tl.store(out + i, x + y, i < n)


def add(a, b, *, out=None, block_size=1024, num_warps=4):
    if out is None:
        out = torch.empty_like(a)
    if a.numel():
        with torch.cuda.device(a.device):
            vector_add[(triton.cdiv(a.numel(), block_size),)](
                a, b, out, a.numel(), BLOCK=block_size, num_warps=num_warps
            )
    return out
