"""JAX baseline for 30 · Sigmoid Linear Unit. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def silu(x):
    raise NotImplementedError("silu: JAX baseline not implemented (jax/implementation.py)")
