"""43 · Attention with Linear Biases: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Causal attention for Q, K, V (B, H, S, D) with ALiBi bias slopes[h] * (j - i)
added to the scaled scores; slopes is FP32 (H).
"""

import sys

import torch

from lab import dispatch

FUNCTION = "alibi_attention"
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


def alibi_attention(q, k, v, slopes, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, q, k, v, slopes, **options)
