"""04 · Interleave Arrays: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

out[2*i] = a[i] and out[2*i + 1] = b[i] for equal-length 1-D a, b.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "interleave"
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


def interleave(a, b, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, a, b, **options)
