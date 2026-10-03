"""25 · General Matrix Multiplication (GEMM) — FP16: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

out = alpha * (A @ B) + beta * C for FP16 A (M, K), B (K, N), C (M, N),
FP32 accumulation, FP16 output. C is not modified.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "gemm"
BACKENDS = ("pytorch", "cuda", "triton", "jax")
DTYPES = {
    "fp16": torch.float16,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.float16: (0.01, 0.01),
}


def gemm(a, b, c, alpha, beta, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, a, b, c, alpha, beta, **options)
