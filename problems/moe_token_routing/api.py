"""50 · MoE Token Routing and Dispatch: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Top-k routing of tokens x (T, D) by FP32 router_logits (T, E). Returns
(dispatched (T*k, D) rows of x grouped by expert ascending and by token within
an expert; expert_offsets int32 (E + 1); token_index int32 (T*k); weights FP32
(T*k), the softmax over each token's k selected logits).
"""

import sys

import torch

from lab import dispatch

FUNCTION = "route_tokens"
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


def route_tokens(x, router_logits, top_k, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(
        sys.modules[__name__], FUNCTION, backend, x, router_logits, top_k, **options
    )
