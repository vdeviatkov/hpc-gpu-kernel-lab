"""JAX baseline for 41 · Causal Self-Attention. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def causal_attention(q, k, v):
    raise NotImplementedError(
        "causal_attention: JAX baseline not implemented (jax/implementation.py)"
    )
