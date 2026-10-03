"""JAX baseline for 43 · Attention with Linear Biases. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def alibi_attention(q, k, v, slopes):
    raise NotImplementedError(
        "alibi_attention: JAX baseline not implemented (jax/implementation.py)"
    )
