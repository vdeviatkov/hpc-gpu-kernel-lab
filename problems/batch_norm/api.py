"""35 · Batch Normalization: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Training-mode batch normalization of x (N, C, H, W): statistics per channel
over N, H, W (biased variance, FP32), then * weight[c] + bias[c].
"""

import sys

import torch

from lab import dispatch

FUNCTION = "batch_norm"
BACKENDS = ("pytorch", "cuda", "triton", "jax")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.float32: (0.0001, 0.0001),
    torch.float16: (0.01, 0.01),
    torch.bfloat16: (0.05, 0.05),
}


def batch_norm(x, weight, bias, eps, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, weight, bias, eps, **options)
