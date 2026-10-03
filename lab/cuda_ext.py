"""Lazy, cached build of a problem's CUDA extension through PyTorch's extension API."""

from functools import cache
from pathlib import Path


@cache
def load_extension(name, directory, sources=("bindings.cpp", "kernels.cu")):
    """Build once per process with O3 and line info; fast math is never enabled."""
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit not found; install nvcc and set CUDA_HOME")
    directory = Path(directory)
    return load(
        name=name,
        sources=[str(directory / source) for source in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
    )
