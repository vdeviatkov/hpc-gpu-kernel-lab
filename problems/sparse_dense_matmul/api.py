"""28 · Sparse Matrix-Dense Matrix Multiplication: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

C = A @ B for CSR A (M, K) given as row_ptr int32 [M + 1], col_idx int32,
values, and dense B (K, N); FP32 accumulation.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "spmm"
BACKENDS = ("pytorch", "cuda", "triton")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.float32: (0.0001, 0.0001),
    torch.float16: (0.01, 0.01),
    torch.bfloat16: (0.05, 0.05),
}


def spmm(row_ptr, col_idx, values, b, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(
        sys.modules[__name__], FUNCTION, backend, row_ptr, col_idx, values, b, **options
    )
