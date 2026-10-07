"""Triton backend for 02 · ReLU. Tile size and warp count are explicit arguments; there
is no autotuning.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    x = tl.load(x_ptr + i, mask=mask)
    # Same formula as the contract: only strictly negative values become 0, so NaN and
    # -0.0 pass through. tl.maximum(x, 0) would not.
    tl.store(out_ptr + i, tl.where(x < 0, tl.zeros_like(x), x), mask=mask)


def relu(x, *, block_size=1024, num_warps=4):
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        with torch.cuda.device(x.device):
            relu_kernel[(triton.cdiv(n, block_size),)](
                x, out, n, BLOCK=block_size, num_warps=num_warps
            )
    return out
