"""20 · Parallel Merge: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Merge sorted 1-D tensors a (M) and b (N) into sorted out (M + N);
stable: on ties, elements of a come first.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "merge"
BACKENDS = ("pytorch", "cuda")
DTYPES = {
    "fp32": torch.float32,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.float32: (0, 0),
}


def merge(a, b, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, a, b, **options)
