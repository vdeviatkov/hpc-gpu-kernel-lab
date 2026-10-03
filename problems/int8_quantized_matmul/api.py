"""27 · INT8 Quantized MatMul: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

out = scale_a * scale_b * ((A - zero_a) @ (B - zero_b)) for int8 A (M, K)
and B (K, N), int32 accumulation, FP32 output. Confirm the source's
requantization and output type before implementing.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "int8_matmul"
BACKENDS = ("pytorch", "cuda", "triton")
DTYPES = {
    "int8": torch.int8,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.int8: (1e-05, 1e-05),
}


def int8_matmul(a, b, scale_a, scale_b, zero_a, zero_b, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(
        sys.modules[__name__], FUNCTION, backend, a, b, scale_a, scale_b, zero_a, zero_b, **options
    )
