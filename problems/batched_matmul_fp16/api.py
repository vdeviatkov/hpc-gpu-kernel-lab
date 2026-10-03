"""26 · FP16 Batched Matrix Multiplication: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

C[i] = A[i] @ B[i] for FP16 A (batch, M, K), B (batch, K, N);
FP32 accumulation, FP16 output.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "bmm"
BACKENDS = ("pytorch", "cuda", "triton", "jax")
DTYPES = {
    "fp16": torch.float16,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.float16: (0.01, 0.01),
}


def bmm(a, b, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, a, b, **options)
