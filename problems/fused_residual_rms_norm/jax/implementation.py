"""JAX baseline for 36 · Fused Residual Add and RMS Norm. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def fused_add_rms_norm(x, residual, weight, eps):
    raise NotImplementedError(
        "fused_add_rms_norm: JAX baseline not implemented (jax/implementation.py)"
    )
