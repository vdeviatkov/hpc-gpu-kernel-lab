"""Build once, lazily; all device launches use PyTorch's current stream."""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit not found; install nvcc and set CUDA_HOME")
    here = Path(__file__).parent
    return load(
        name="gpu_lab_vector_add",
        sources=[str(here / "bindings.cpp"), str(here / "kernels.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
    )


def add(a, b, *, out=None, variant=0, threads=256):
    if out is None:
        out = torch.empty_like(a)
    extension().add_out(a, b, out, variant, threads)
    return out
