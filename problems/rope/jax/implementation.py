"""JAX baseline for 38 · Rotary Positional Embedding. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def rope(x, cos, sin):
    raise NotImplementedError("rope: JAX baseline not implemented (jax/implementation.py)")
