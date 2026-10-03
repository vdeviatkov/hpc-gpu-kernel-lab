"""Triton backend for 15 · Mean Squared Error. Not implemented yet.

Implement mse_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def mse_kernel(pred_ptr, target_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def mse(pred, target, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError("mse: Triton backend not implemented (triton/implementation.py)")
    out = torch.empty((), dtype=torch.float32, device=pred.device)
    n = pred.numel()
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(pred.device):
        mse_kernel[grid](pred, target, out, n, BLOCK=block_size, num_warps=num_warps)
    return out
