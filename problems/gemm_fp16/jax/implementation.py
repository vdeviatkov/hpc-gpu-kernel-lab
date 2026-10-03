"""JAX baseline for 25 · General Matrix Multiplication (GEMM) — FP16. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def gemm(a, b, c, alpha, beta):
    raise NotImplementedError("gemm: JAX baseline not implemented (jax/implementation.py)")
