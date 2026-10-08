"""Lazy, cached build of a problem's CUDA extension through PyTorch's extension API."""

from functools import cache
from pathlib import Path

# Shared headers for every extension: #include "lab/vector.cuh", "lab/launch.cuh",
# "lab/checks.h". Ninja tracks header dependencies, so editing them triggers a rebuild.
INCLUDE_DIR = Path(__file__).parent / "cuda" / "include"


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
        extra_include_paths=[str(INCLUDE_DIR)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
    )
