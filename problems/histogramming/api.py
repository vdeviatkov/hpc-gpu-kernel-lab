"""18 · Histogramming: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Count occurrences of each value in [0, bins) for a 1-D int32 tensor;
returns int32 counts of length bins. Inputs are guaranteed in range.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "histogram"
BACKENDS = ("pytorch", "cuda", "triton")
DTYPES = {
    "int32": torch.int32,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.int32: (0, 0),
}


def histogram(x, bins, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, bins, **options)
