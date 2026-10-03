"""JAX baseline for 33 · RMS Normalization. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def rms_norm(x, weight, eps):
    raise NotImplementedError("rms_norm: JAX baseline not implemented (jax/implementation.py)")
