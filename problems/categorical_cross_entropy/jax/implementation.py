"""JAX baseline for 37 · Categorical Cross Entropy Loss. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def cross_entropy(logits, labels):
    raise NotImplementedError("cross_entropy: JAX baseline not implemented (jax/implementation.py)")
