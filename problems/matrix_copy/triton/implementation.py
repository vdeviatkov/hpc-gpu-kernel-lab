"""Triton copies over flat 1-D blocks and over (BLOCK_M, BLOCK_N) tiles.

Both kernels move integers of the element's width (the tensors are viewed as int32 or
int16), so no value passes through a floating-point register and bit patterns are copied
unchanged. Block sizes and warp counts are explicit; there is no autotuning. The default
64 x 64 tile is the one 08 · Matrix Transpose starts from, so the two compare one tile.
"""

import torch
import triton
import triton.language as tl

BITS = {4: torch.int32, 2: torch.int16}
# Grid axis 1 is limited to 65535 programs.
MAX_GRID_Y = 65535


@triton.jit
def flat_kernel(a, out, total, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < total
    tl.store(out + i, tl.load(a + i, mask=mask), mask=mask)


@triton.jit
def tile_kernel(a, out, m, n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(1).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * n + cols[None, :]
    mask = (rows[:, None] < m) & (cols[None, :] < n)
    tl.store(out + offsets, tl.load(a + offsets, mask=mask), mask=mask)


def _power_of_two(value):
    return value > 0 and not value & (value - 1)


def matrix_copy(a, *, layout="flat", block_size=1024, block_m=64, block_n=64, num_warps=4):
    if layout not in ("flat", "2d"):
        raise ValueError("layout must be 'flat' or '2d'")
    if not all(_power_of_two(b) for b in (block_size, block_m, block_n)):
        raise ValueError("block sizes must be positive powers of two")
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    if not a.numel():
        return out
    m, n = a.shape
    src, dst = a.view(BITS[a.element_size()]), out.view(BITS[a.element_size()])
    with torch.cuda.device(a.device):
        if layout == "flat":
            grid = (triton.cdiv(a.numel(), block_size),)
            flat_kernel[grid](src, dst, a.numel(), BLOCK=block_size, num_warps=num_warps)
        else:
            grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))
            if grid[1] > MAX_GRID_Y:
                raise ValueError(f"{m} rows need more than {MAX_GRID_Y} row tiles")
            tile_kernel[grid](src, dst, m, n, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps)
    return out
