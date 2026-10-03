"""JAX baseline for 39 · SwiGLU MLP Block. Not implemented yet.

Receives JAX arrays on the same device as the PyTorch inputs (through DLPack) and
returns JAX arrays. Compile with jax.jit; the caller waits for completion.
"""


def swiglu_mlp(x, w_gate, w_up, w_down):
    raise NotImplementedError("swiglu_mlp: JAX baseline not implemented (jax/implementation.py)")
