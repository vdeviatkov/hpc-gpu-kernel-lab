"""JAX baseline for 24 · Batched Matrix Multiplication — FP32. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def bmm(a, b):
    raise NotImplementedError("bmm: JAX baseline not implemented (jax/implementation.py)")
