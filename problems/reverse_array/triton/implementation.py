"""Triton backends for 03 · Reverse Array. Tile size and warp count are explicit arguments;
there is no autotuning.

Each program owns BLOCK pairs: front elements [f, f + len) and their mirrors. Triton, not
the programmer, decides which thread handles which tile element, and tl.flip moves values
between threads, so a thread could store to an address another thread has not read yet.
A block-wide barrier (tl.debug_barrier, i.e. __syncthreads) between the loads and the
stores rules that out, the same role __syncthreads plays in the CUDA tile kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def reverse_pairs_kernel(x_ptr, n, BLOCK: tl.constexpr):
    """Element t owns pair (i, n - 1 - i); the back half is addressed in descending order."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n // 2
    j = n - 1 - i
    front = tl.load(x_ptr + i, mask=mask)
    back = tl.load(x_ptr + j, mask=mask)
    tl.debug_barrier()
    tl.store(x_ptr + i, back, mask=mask)
    tl.store(x_ptr + j, front, mask=mask)


@triton.jit
def reverse_flip_kernel(x_ptr, n, BLOCK: tl.constexpr):
    """Both halves are loaded in ascending order; tl.flip reverses them in registers.

    The back tile starts at n - f - BLOCK, so element t of it mirrors front element
    BLOCK - 1 - t; only its last `len` elements belong to this program.
    """
    front_start = tl.program_id(0).to(tl.int64) * BLOCK
    length = tl.minimum(n // 2 - front_start, BLOCK)
    t = tl.arange(0, BLOCK)
    front_offsets = front_start + t
    back_offsets = n - front_start - BLOCK + t
    front_mask = t < length
    back_mask = t >= BLOCK - length
    front = tl.load(x_ptr + front_offsets, mask=front_mask)
    back = tl.load(x_ptr + back_offsets, mask=back_mask)
    tl.debug_barrier()
    tl.store(x_ptr + front_offsets, tl.flip(back, 0), mask=front_mask)
    tl.store(x_ptr + back_offsets, tl.flip(front, 0), mask=back_mask)


def reverse_(x, *, flip=False, block_size=1024, num_warps=4):
    n = x.numel()
    if n >= 2:
        kernel = reverse_flip_kernel if flip else reverse_pairs_kernel
        with torch.cuda.device(x.device):
            kernel[(triton.cdiv(n // 2, block_size),)](x, n, BLOCK=block_size, num_warps=num_warps)
    return x
