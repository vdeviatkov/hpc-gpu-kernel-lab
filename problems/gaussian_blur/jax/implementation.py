"""JAX baseline for 10 · Gaussian Blur. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def gaussian_blur(image, kernel):
    raise NotImplementedError("gaussian_blur: JAX baseline not implemented (jax/implementation.py)")
