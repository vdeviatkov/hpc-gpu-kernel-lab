"""02 · ReLU: public entry point.

Contract, verified against the LeetGPU statement (challenges/easy/21_relu):

    out[i] = max(0, x[i])   for 0 <= i < N

- Source: contiguous 1-D FP32, 1 <= N <= 100,000,000; the performance test uses
  N = 25,000,000 with inputs uniform in [-100, 100]; reference torch.relu.
- Lab extensions: FP16/BF16 and N = 0.
- Results match torch.relu exactly: only strictly negative values become 0, so the
  kernel formula is `x < 0 ? 0 : x`.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "relu"
# CUDA variants differ only in how the condition is written, or in the load width.
VARIANTS = {
    "cuda_select": ("cuda", {"variant": 0}),  # x < 0 ? 0 : x
    "cuda_branch": ("cuda", {"variant": 1}),  # if/else
    "cuda_fmax": ("cuda", {"variant": 2}),  # fmaxf(x, 0): differs on NaN and -0.0
    "cuda_vec": ("cuda", {"variant": 3}),  # select with 16-byte loads and stores
}
BACKENDS = ("pytorch", *VARIANTS, "triton")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference. ReLU computes no new values, so every
# backend must match bit for bit.
TOLERANCES = {
    torch.float32: (0, 0),
    torch.float16: (0, 0),
    torch.bfloat16: (0, 0),
}


def relu(x, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected torch.Tensor")
    if x.layout != torch.strided or x.ndim != 1 or not x.is_contiguous():
        raise ValueError("expected a contiguous one-dimensional tensor")
    if x.dtype not in DTYPES.values():
        raise ValueError("supported dtypes: fp32, fp16, bf16")
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, **options)
