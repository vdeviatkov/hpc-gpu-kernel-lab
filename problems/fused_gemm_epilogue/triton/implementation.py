"""Triton backend for 29 · Fused GEMM + Bias + Activation. Not implemented yet.

Implement linear_relu_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def linear_relu_kernel(x_ptr, w_ptr, bias_ptr, out_ptr, m, k, n, BLOCK: tl.constexpr):
    pass  # TODO: implement.


def linear_relu(x, w, bias, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "linear_relu: Triton backend not implemented (triton/implementation.py)"
        )
    out = torch.empty((x.shape[0], w.shape[1]), dtype=x.dtype, device=x.device)
    m = x.shape[0]
    k = x.shape[1]
    n = w.shape[1]
    grid = (triton.cdiv(m, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        linear_relu_kernel[grid](x, w, bias, out, m, k, n, BLOCK=block_size, num_warps=num_warps)
    return out
