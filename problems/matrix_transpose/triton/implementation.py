"""Triton backend for 08 · Matrix Transpose. Not implemented yet.

Implement transpose_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def transpose_kernel(a_ptr, out_ptr, m, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def transpose(a, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "transpose: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty((a.shape[1], a.shape[0]), dtype=a.dtype, device=a.device)
    m = a.shape[0]
    n = a.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(a.device):
        transpose_kernel[grid](a, out, m, n, BLOCK=block_size, num_warps=num_warps)
    return out
