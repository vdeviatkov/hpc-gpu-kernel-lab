"""Route an api-level call to a backend implementation module.

Backend modules live at problems/<problem>/<backend>/implementation.py and define a
function with the same name as the api function. They are imported only when
selected, so optional compilers (nvcc, Triton, JAX) are never required up front.
"""

from importlib import import_module

import torch

GPU_BACKENDS = ("cuda", "triton")


def _tensors(args):
    return [a for a in args if isinstance(a, torch.Tensor)]


def _to_jax(value):
    if isinstance(value, torch.Tensor):
        import jax

        return jax.dlpack.from_dlpack(value)
    return value


def _from_jax(value):
    if isinstance(value, tuple | list):
        return type(value)(_from_jax(v) for v in value)
    if hasattr(value, "block_until_ready"):
        # JAX runs on its own stream: wait for completion before PyTorch reads the data.
        return torch.from_dlpack(value.block_until_ready())
    return value


def call(api, function, backend, *args, **options):
    """Validate the backend name and device placement, then call the implementation."""
    if backend not in api.BACKENDS:
        raise ValueError(f"unknown backend: {backend}; choose from {', '.join(api.BACKENDS)}")
    tensors = _tensors(args)
    if backend in GPU_BACKENDS and any(
        t.device.type != "cuda" or torch.version.cuda is None for t in tensors
    ):
        raise ValueError(f"{backend} backend requires NVIDIA CUDA tensors")
    module = import_module(f"{api.__name__.rpartition('.')[0]}.{backend}.implementation")
    fn = getattr(module, function)
    if backend == "jax":
        return _from_jax(fn(*(_to_jax(a) for a in args), **options))
    return fn(*args, **options)
