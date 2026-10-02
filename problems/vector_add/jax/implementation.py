"""Native JAX arrays; callers explicitly place data and wait for completion."""

import jax
import jax.numpy as jnp


@jax.jit
def _add(a, b):
    # Make low-precision arithmetic intent match CUDA/Triton explicitly.
    return (a.astype(jnp.float32) + b.astype(jnp.float32)).astype(a.dtype)


def add(a, b):
    if a.ndim != 1 or a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("expected matching 1-D shapes and dtypes")
    if a.dtype not in (jnp.float32, jnp.float16, jnp.bfloat16):
        raise ValueError("supported dtypes: fp32, fp16, bf16")
    if len(a.devices()) != 1 or a.devices() != b.devices():
        raise ValueError("inputs must share one device")
    return _add(a, b)
