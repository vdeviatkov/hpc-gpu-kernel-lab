"""JAX baseline for 13 · Reduction. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def reduce_sum(x):
    raise NotImplementedError("reduce_sum: JAX baseline not implemented (jax/implementation.py)")
