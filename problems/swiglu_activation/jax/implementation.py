"""JAX baseline for 31 · Swish-Gated Linear Unit. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def swiglu(x):
    raise NotImplementedError("swiglu: JAX baseline not implemented (jax/implementation.py)")
