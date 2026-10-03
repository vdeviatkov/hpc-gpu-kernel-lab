"""JAX baseline for 44 · Grouped Query Attention. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def gqa(q, k, v):
    raise NotImplementedError("gqa: JAX baseline not implemented (jax/implementation.py)")
