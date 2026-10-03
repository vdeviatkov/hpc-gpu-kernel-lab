"""14 · Count Array Element: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Number of elements of the 1-D int32 tensor x equal to value; 0-d int32 result.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "count_equal"
BACKENDS = ("pytorch", "cuda", "triton")
DTYPES = {
    "int32": torch.int32,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.int32: (0, 0),
}


def count_equal(x, value, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, value, **options)
