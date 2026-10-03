"""17 · Prefix Sum: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Inclusive prefix sum of a 1-D FP32 tensor: out[i] = x[0] + ... + x[i].
"""

import sys

import torch

from lab import dispatch

FUNCTION = "inclusive_scan"
BACKENDS = ("pytorch", "cuda", "triton")
DTYPES = {
    "fp32": torch.float32,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.float32: (0.0001, 0.01),
}


def inclusive_scan(x, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, **options)
