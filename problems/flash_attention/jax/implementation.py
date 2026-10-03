"""JAX baseline for 48 · FlashAttention-Style Online Attention. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def flash_attention(q, k, v, causal):
    raise NotImplementedError(
        "flash_attention: JAX baseline not implemented (jax/implementation.py)"
    )
