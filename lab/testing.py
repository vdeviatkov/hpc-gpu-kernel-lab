"""Test helpers: availability skips and the backend-vs-reference check."""

import importlib.util

import pytest
import torch

from lab import dispatch
from lab.data import Sampler, clone_tensors

CUDA = torch.cuda.is_available() and torch.version.cuda is not None


def device():
    return "cuda" if CUDA else "cpu"


def skip_if_unavailable(backend):
    """Skip explicitly when hardware or an optional package is missing."""
    if backend in dispatch.GPU_BACKENDS and not CUDA:
        pytest.skip("NVIDIA GPU unavailable")
    if backend == "triton" and importlib.util.find_spec("triton") is None:
        pytest.skip("optional Triton package unavailable")
    if backend == "jax" and importlib.util.find_spec("jax") is None:
        pytest.skip("optional JAX package unavailable")
    if backend == "cuda":
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            pytest.skip("CUDA toolkit unavailable")


def skip_unsupported_dtype(dtype):
    if CUDA and dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 test policy requires SM80+")


def call_or_skip(api, backend, args):
    try:
        return dispatch.call(api, api.FUNCTION, backend, *args)
    except NotImplementedError as error:
        pytest.skip(f"not implemented: {error}")


def check_against_reference(api, cases, backend, case, dtype_name, seed=0):
    """Compare a backend with the PyTorch reference on identical inputs.

    Inputs are cloned per call; unless the problem declares MUTATES, the inputs
    must also come back unchanged.
    """
    skip_if_unavailable(backend)
    dtype = api.DTYPES[dtype_name]
    skip_unsupported_dtype(dtype)
    inputs = cases.make_inputs(case, Sampler(dtype, device(), seed))
    expected = call_or_skip(api, "pytorch", clone_tensors(inputs))
    args = clone_tensors(inputs)
    actual = call_or_skip(api, backend, args)
    rtol, atol = api.TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    if not getattr(api, "MUTATES", False):
        for before, after in zip(inputs, args, strict=True):
            if isinstance(before, torch.Tensor):
                torch.testing.assert_close(after, before, rtol=0, atol=0, equal_nan=True)


def case_id(case):
    """Readable pytest id for a case dict, e.g. m=3,n=5."""
    return ",".join(f"{k}={v}" for k, v in case.items())
