"""PyTorch reference and eager baseline for 49 · Paged Attention.

Not implemented yet. Every other backend is tested against this function, so
write the plainest correct version of the contract in api.py.
"""


def paged_attention(q, k_cache, v_cache, block_table, context_lens):
    raise NotImplementedError(
        "paged_attention: PyTorch reference not implemented (pytorch/implementation.py)"
    )
