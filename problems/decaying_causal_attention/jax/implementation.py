"""JAX baseline for 45 · Decaying Causal Attention. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def decaying_attention(q, k, v, gamma):
    raise NotImplementedError(
        "decaying_attention: JAX baseline not implemented (jax/implementation.py)"
    )
