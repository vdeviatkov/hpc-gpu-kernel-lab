"""46 · Quantize / Dequantize Pipeline: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Symmetric per-group int8 quantization of x (M, N) along N with group_size
dividing N: scale = max|x_group| / 127 (FP32), q = clamp(round(x / scale),
-127, 127). Returns (q int8 (M, N), scale FP32 (M, N / group_size)).
"""

import sys

import torch

from lab import dispatch

FUNCTION = "quantize"
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


def quantize(x, group_size, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, group_size, **options)
