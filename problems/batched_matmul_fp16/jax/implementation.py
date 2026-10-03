"""JAX baseline for 26 · FP16 Batched Matrix Multiplication. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def bmm(a, b):
    raise NotImplementedError("bmm: JAX baseline not implemented (jax/implementation.py)")
