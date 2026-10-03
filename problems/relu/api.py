"""02 · ReLU: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

out[i] = max(x[i], 0) for a contiguous 1-D tensor x.
Policy for NaN and -0.0 must be stated before implementing.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "relu"
BACKENDS = ("pytorch", "cuda", "triton")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.float32: (0, 0),
    torch.float16: (0, 0),
    torch.bfloat16: (0, 0),
}


def relu(x, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, **options)
