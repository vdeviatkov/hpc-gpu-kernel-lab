"""11 · 2D Max Pooling: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

2-D max pooling over an (N, C, H, W) tensor with square kernel_size,
stride and zero-based padding (padded positions never win).
"""

import sys

import torch

from lab import dispatch

FUNCTION = "max_pool2d"
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


def max_pool2d(x, kernel_size, stride, padding, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(
        sys.modules[__name__], FUNCTION, backend, x, kernel_size, stride, padding, **options
    )
