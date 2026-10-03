"""JAX baseline for 09 · 1D Convolution. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def conv1d(x, w):
    raise NotImplementedError("conv1d: JAX baseline not implemented (jax/implementation.py)")
