"""One fused RGB-to-grayscale kernel, with FP32 arithmetic for every input dtype.

Each program owns BLOCK pixels of the flattened image, independent of row shape.
Tile size and warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rgb_to_grayscale_kernel(image_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    r = tl.load(image_ptr + 3 * i, mask=i < n, other=0).to(tl.float32)
    g = tl.load(image_ptr + 3 * i + 1, mask=i < n, other=0).to(tl.float32)
    b = tl.load(image_ptr + 3 * i + 2, mask=i < n, other=0).to(tl.float32)
    gray = r * 0.299 + g * 0.587 + b * 0.114
    tl.store(out_ptr + i, gray, mask=i < n)


def rgb_to_grayscale(image, *, block_size=1024, num_warps=4):
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    out = torch.empty(image.shape[:2], dtype=image.dtype, device=image.device)
    n = image.numel() // 3
    if n:
        with torch.cuda.device(image.device):
            rgb_to_grayscale_kernel[(triton.cdiv(n, block_size),)](
                image, out, n, BLOCK=block_size, num_warps=num_warps
            )
    return out
