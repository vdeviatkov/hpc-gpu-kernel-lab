"""Triton backend for 10 · Gaussian Blur. Not implemented yet.

Implement gaussian_blur_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def gaussian_blur_kernel(image_ptr, kernel_ptr, out_ptr, h, w, k, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def gaussian_blur(image, kernel, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "gaussian_blur: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty_like(image)
    h = image.shape[0]
    w = image.shape[1]
    k = kernel.shape[0]
    grid = (triton.cdiv(h, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(image.device):
        gaussian_blur_kernel[grid](
            image, kernel, out, h, w, k, BLOCK=block_size, num_warps=num_warps
        )
    return out
