"""36 · Fused Residual Add and RMS Norm: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

residual_out = x + residual; out = rms_norm(residual_out) * weight over the
last dimension of (M, N) tensors. Returns (out, residual_out).
"""

import sys

import torch

from lab import dispatch

FUNCTION = "fused_add_rms_norm"
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


def fused_add_rms_norm(x, residual, weight, eps, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(
        sys.modules[__name__], FUNCTION, backend, x, residual, weight, eps, **options
    )
