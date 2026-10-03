"""JAX baseline for 42 · Multi-Head Attention. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def mha(q, k, v):
    raise NotImplementedError("mha: JAX baseline not implemented (jax/implementation.py)")
