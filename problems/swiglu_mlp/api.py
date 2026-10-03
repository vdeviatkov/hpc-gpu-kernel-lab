"""39 · SwiGLU MLP Block: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

out = (silu(X @ W_gate) * (X @ W_up)) @ W_down for X (M, D), W_gate and W_up
(D, F), W_down (F, D); FP32 accumulation.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "swiglu_mlp"
BACKENDS = ("pytorch", "cuda", "triton", "jax")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.float32: (0.0001, 0.001),
    torch.float16: (0.01, 0.01),
    torch.bfloat16: (0.05, 0.05),
}


def swiglu_mlp(x, w_gate, w_up, w_down, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(
        sys.modules[__name__], FUNCTION, backend, x, w_gate, w_up, w_down, **options
    )
