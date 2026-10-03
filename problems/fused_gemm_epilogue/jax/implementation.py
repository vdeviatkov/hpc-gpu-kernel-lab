"""JAX baseline for 29 · Fused GEMM + Bias + Activation. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def linear_relu(x, w, bias):
    raise NotImplementedError("linear_relu: JAX baseline not implemented (jax/implementation.py)")
