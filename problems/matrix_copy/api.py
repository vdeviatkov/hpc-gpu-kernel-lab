"""07 · Matrix Copy: public entry point.

Contract, verified against the LeetGPU statement (challenges/easy/31_matrix_copy):

    B[i, j] = A[i, j]   for every 0 <= i, j < N

- Source: an N x N row-major float32 matrix A copied into B; 1 <= N <= 4096. The
  performance test is N = 4096. The source checker allows atol = rtol = 1e-5; a copy has
  no arithmetic, so here every backend must reproduce the input bit for bit (NaN payloads
  and signed zeros included).
- Here: a contiguous 2-D tensor in, a new contiguous tensor of the same shape, dtype and
  device out. The input is never modified.
- Lab extensions: rectangular M x N matrices, empty matrices (M = 0 or N = 0), N > 4096,
  and fp16/bf16 elements. Strided (pitched) layouts are not accepted.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "matrix_copy"
BACKENDS = ("pytorch", "cuda", "triton")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# A copy must be exact. assert_close treats -0.0 == 0.0 and any NaN == NaN, so the tests
# also compare bit patterns.
TOLERANCES = {
    torch.float32: (0, 0),
    torch.float16: (0, 0),
    torch.bfloat16: (0, 0),
}


def matrix_copy(a, *, backend="pytorch", **options):
    """Return a new contiguous tensor equal to the 2-D matrix `a`, bit for bit."""
    if not isinstance(a, torch.Tensor):
        raise TypeError("expected torch.Tensor")
    if a.layout != torch.strided or a.ndim != 2 or not a.is_contiguous():
        raise ValueError("expected a contiguous two-dimensional tensor")
    if a.dtype not in DTYPES.values():
        raise ValueError("supported dtypes: fp32, fp16, bf16")
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, a, **options)
