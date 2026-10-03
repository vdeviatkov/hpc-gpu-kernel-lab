"""19 · Stream Compaction: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Return the elements x[i] > 0 of a 1-D tensor in their original order
(stable). The output length depends on the data.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "compact_positive"
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


def compact_positive(x, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, **options)
