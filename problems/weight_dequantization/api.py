"""12 · Weight Dequantization: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

out[i, j] = q[i, j] * scale[i // quant_block, j // quant_block] for int8 q (M, N)
and per-block scales; out has the scale dtype.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "dequantize"
BACKENDS = ("pytorch", "cuda", "triton")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.float32: (1e-05, 1e-05),
    torch.float16: (0.002, 0.002),
    torch.bfloat16: (0.016, 0.016),
}


def dequantize(q, scale, quant_block, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, q, scale, quant_block, **options)
