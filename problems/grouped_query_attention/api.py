"""44 · Grouped Query Attention: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Causal attention with Q (B, Hq, S, D) and K, V (B, Hkv, S, D), Hq % Hkv == 0:
query head h uses key/value head h // (Hq / Hkv). Scale 1/sqrt(D).
"""

import sys

import torch

from lab import dispatch

FUNCTION = "gqa"
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


def gqa(q, k, v, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, q, k, v, **options)
