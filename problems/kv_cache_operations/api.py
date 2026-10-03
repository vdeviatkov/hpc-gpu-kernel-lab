"""47 · KV-Cache Update and Paged Access: public entry point.

Draft contract, to confirm against the source statement linked in README.md before
implementing:

Write new keys/values k_new, v_new (B, H, T, D) into paged caches k_cache,
v_cache (num_blocks, H, block_size, D) at token positions positions[b] ..
positions[b] + T - 1, where block_table[b, p // block_size] gives the physical
block. Caches are updated in place; returns (k_cache, v_cache).
"""

import sys

import torch

from lab import dispatch

FUNCTION = "kv_cache_append"
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

# The function updates its inputs in place.
MUTATES = True


def kv_cache_append(
    k_new, v_new, k_cache, v_cache, block_table, positions, *, backend="pytorch", **options
):
    """See the module docstring. `options` are backend launch parameters."""
    # TODO: validate shapes, dtypes, devices and layout per the contract
    # (see problems/vector_add/api.py for a complete example).
    return dispatch.call(
        sys.modules[__name__],
        FUNCTION,
        backend,
        k_new,
        v_new,
        k_cache,
        v_cache,
        block_table,
        positions,
        **options,
    )
