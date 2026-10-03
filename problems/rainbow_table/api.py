"""06 · Rainbow Table: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Apply the source's 32-bit hash `rounds` times to every element of x.
Values are uint32 bit patterns stored in int32 tensors; the exact hash
and overflow behavior must be copied from the source statement.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "rainbow_table"
BACKENDS = ("pytorch", "cuda")
DTYPES = {
    "int32": torch.int32,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.int32: (0, 0),
}


def rainbow_table(x, rounds, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, rounds, **options)
