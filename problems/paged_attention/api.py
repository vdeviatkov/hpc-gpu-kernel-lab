"""49 · Paged Attention: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Decode attention: one query q[b] (B, H, D) per sequence attends to its first
context_lens[b] cached tokens, stored in paged k_cache/v_cache (num_blocks, H,
block_size, D) via block_table[b, p // block_size]. Scale 1/sqrt(D).
"""

import sys

import torch

from lab import dispatch

FUNCTION = "paged_attention"
BACKENDS = ("pytorch", "cuda", "triton")
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


def paged_attention(
    q, k_cache, v_cache, block_table, context_lens, *, backend="pytorch", **options
):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(
        sys.modules[__name__],
        FUNCTION,
        backend,
        q,
        k_cache,
        v_cache,
        block_table,
        context_lens,
        **options,
    )
