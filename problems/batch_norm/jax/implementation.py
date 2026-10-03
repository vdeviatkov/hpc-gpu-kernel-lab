"""JAX baseline for 35 · Batch Normalization. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def batch_norm(x, weight, bias, eps):
    raise NotImplementedError("batch_norm: JAX baseline not implemented (jax/implementation.py)")
