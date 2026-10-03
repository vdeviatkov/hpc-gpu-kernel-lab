"""23 · Matrix Multiplication: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

C = A @ B for contiguous FP32 A (M, K) and B (K, N); no TF32.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "matmul"
BACKENDS = ("pytorch", "cuda", "triton", "jax")
DTYPES = {
    "fp32": torch.float32,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.float32: (0.0001, 0.001),
}


def matmul(a, b, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, a, b, **options)
