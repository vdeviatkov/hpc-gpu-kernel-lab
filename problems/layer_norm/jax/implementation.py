"""JAX baseline for 34 · Layer Normalization. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def layer_norm(x, weight, bias, eps):
    raise NotImplementedError("layer_norm: JAX baseline not implemented (jax/implementation.py)")
