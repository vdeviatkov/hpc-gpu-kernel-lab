"""JAX baseline for 32 · Softmax. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def softmax(x):
    raise NotImplementedError("softmax: JAX baseline not implemented (jax/implementation.py)")
