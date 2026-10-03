"""JAX baseline for 21 · Dense GEMV. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def gemv(a, x):
    raise NotImplementedError("gemv: JAX baseline not implemented (jax/implementation.py)")
