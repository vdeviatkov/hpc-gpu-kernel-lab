"""Triton backend for 11 · 2D Max Pooling. Not implemented yet.

Implement max_pool2d_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def max_pool2d_kernel(
    x_ptr, kernel_size, stride, padding, out_ptr, n, c, h, w, BLOCK: tl.constexpr
):
    pass  # TODO: implement.


def max_pool2d(x, kernel_size, stride, padding, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "max_pool2d: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty(
        (
            *x.shape[:2],
            (x.shape[2] + 2 * padding - kernel_size) // stride + 1,
            (x.shape[3] + 2 * padding - kernel_size) // stride + 1,
        ),
        dtype=x.dtype,
        device=x.device,
    )
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    grid = (triton.cdiv(n, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        max_pool2d_kernel[grid](
            x, kernel_size, stride, padding, out, n, c, h, w, BLOCK=block_size, num_warps=num_warps
        )
    return out
