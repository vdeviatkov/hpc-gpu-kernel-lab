"""JAX baseline for 15 · Mean Squared Error. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def mse(pred, target):
    raise NotImplementedError("mse: JAX baseline not implemented (jax/implementation.py)")
