"""37 · Categorical Cross Entropy Loss: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Mean over rows of -log_softmax(logits)[i, labels[i]] for logits (M, C) and
int64 labels (M); FP32 math, 0-d FP32 result.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "cross_entropy"
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


def cross_entropy(logits, labels, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, logits, labels, **options)
