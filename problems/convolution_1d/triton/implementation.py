"""Triton direct 1-D correlation: one program per BLOCK consecutive outputs.

Each program keeps BLOCK FP32 accumulators and walks the K filter taps. Per tap it loads
one w element (the same address for every lane) and the window x[offs + j], then does one
FMA per output. This is the naive per-output stencil from docs/design.md: neighbouring
programs re-read each x element up to K times, mostly from L1/L2. The result is rounded
once to the input dtype. Tile size and warp count are explicit arguments; there is no
autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = True


@triton.jit
def conv1d_kernel(x_ptr, w_ptr, out_ptr, n, k, BLOCK: tl.constexpr):
    out_n = n - k + 1
    offs = tl.program_id(axis=0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < out_n

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in range(k):
        w = tl.load(w_ptr + j).to(tl.float32)
        # offs < out_n implies offs + j <= n - 1, so the output mask also bounds x.
        x = tl.load(x_ptr + offs + j, mask=mask, other=0.0, eviction_policy="evict_last")
        acc += x.to(tl.float32) * w

    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _power_of_two(value):
    return value > 0 and not value & (value - 1)


def conv1d(x, w, *, block_size=1024, num_warps=4):
    if not _power_of_two(block_size):
        raise ValueError("block_size must be a positive power of two")
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    n, k = x.numel(), w.numel()
    out = torch.empty(n - k + 1, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n - k + 1, block_size),)
    with torch.cuda.device(x.device):
        conv1d_kernel[grid](x, w, out, n, k, BLOCK=block_size, num_warps=num_warps)
    return out
