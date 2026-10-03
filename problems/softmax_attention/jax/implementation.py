"""JAX baseline for 40 · Softmax Attention. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def attention(q, k, v):
    raise NotImplementedError("attention: JAX baseline not implemented (jax/implementation.py)")
