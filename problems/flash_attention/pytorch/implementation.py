"""PyTorch reference and eager baseline for 48 · FlashAttention-Style Online Attention.

Not implemented yet. Every other backend is tested against this function, so
write the plainest correct version of the contract in api.py.
"""


def flash_attention(q, k, v, causal):
    raise NotImplementedError(
        "flash_attention: PyTorch reference not implemented (pytorch/implementation.py)"
    )
