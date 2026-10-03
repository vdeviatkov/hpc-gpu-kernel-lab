"""09 · 1D Convolution: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Valid 1-D correlation: out[i] = sum_j x[i + j] * w[j] for i < N - K + 1,
accumulated in FP32.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "conv1d"
BACKENDS = ("pytorch", "cuda", "triton", "jax")
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


def conv1d(x, w, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, w, **options)
