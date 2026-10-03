"""Triton backend for 37 · Categorical Cross Entropy Loss. Not implemented yet.

Implement cross_entropy_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def cross_entropy_kernel(logits_ptr, labels_ptr, out_ptr, m, c, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def cross_entropy(logits, labels, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "cross_entropy: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty((), dtype=torch.float32, device=logits.device)
    m = logits.shape[0]
    c = logits.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(logits.device):
        cross_entropy_kernel[grid](logits, labels, out, m, c, BLOCK=block_size, num_warps=num_warps)
    return out
