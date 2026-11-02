"""Triton transpose over (BLOCK_M, BLOCK_N) input tiles.

Each program loads one tile indexed [input column, input row], which is already the output
tile's [row, column] order, so it is stored without an explicit tl.trans; the compiler
inserts the layout conversion (through shared memory) that lets both sides coalesce.
Tensors are viewed as int32 or int16, so every bit pattern is moved unchanged. Tile size
and warp count are explicit arguments; there is no autotuning. The default 64 x 64 tile
matches 07 · Matrix Copy's 2-D Triton copy, so the two compare one tile.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = True

BITS = {4: torch.int32, 2: torch.int16}
# Grid axis 1 is limited to 65535 programs.
MAX_GRID_Y = 65535


@triton.jit
def transpose_kernel(a_ptr, out_ptr, m, n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # i: input columns = output rows; j: input rows = output columns.
    i = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    j = tl.program_id(1).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = (i[:, None] < n) & (j[None, :] < m)
    # tile[i, j] = a[j, i] = out[i, j]
    tile = tl.load(a_ptr + j[None, :] * n + i[:, None], mask=mask)
    tl.store(out_ptr + i[:, None] * m + j[None, :], tile, mask=mask)


def _power_of_two(value):
    return value > 0 and not value & (value - 1)


def transpose(a, *, block_m=64, block_n=64, num_warps=4):
    if not all(_power_of_two(b) for b in (block_m, block_n)):
        raise ValueError("block sizes must be positive powers of two")
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    m, n = a.shape
    out = torch.empty((n, m), dtype=a.dtype, device=a.device)
    if not a.numel():
        return out
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))
    if grid[1] > MAX_GRID_Y:
        raise ValueError(f"{m} rows need more than {MAX_GRID_Y} row tiles")
    src, dst = a.view(BITS[a.element_size()]), out.view(BITS[a.element_size()])
    with torch.cuda.device(a.device):
        transpose_kernel[grid](
            src, dst, m, n, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps
        )
    return out
