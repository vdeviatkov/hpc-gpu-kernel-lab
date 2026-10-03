"""JAX baseline for 16 · Dot Product. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def dot(a, b):
    raise NotImplementedError("dot: JAX baseline not implemented (jax/implementation.py)")
