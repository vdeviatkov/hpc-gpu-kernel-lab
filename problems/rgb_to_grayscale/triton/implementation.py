"""Triton backend for 05 · RGB to Grayscale. Not implemented yet.

Implement rgb_to_grayscale_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def rgb_to_grayscale_kernel(image_ptr, out_ptr, h, w, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def rgb_to_grayscale(image, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "rgb_to_grayscale: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty(image.shape[:2], dtype=image.dtype, device=image.device)
    h = image.shape[0]
    w = image.shape[1]
    grid = (triton.cdiv(h, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(image.device):
        rgb_to_grayscale_kernel[grid](image, out, h, w, BLOCK=block_size, num_warps=num_warps)
    return out
