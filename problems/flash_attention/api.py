"""48 · FlashAttention-Style Online Attention: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Attention for Q, K, V (B, H, S, D), optionally causal, computed with an
online (streaming) softmax so the (S, S) score matrix is never stored.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "flash_attention"
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


def flash_attention(q, k, v, causal, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, q, k, v, causal, **options)
