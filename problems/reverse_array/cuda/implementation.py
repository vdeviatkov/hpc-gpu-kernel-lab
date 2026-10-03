"""CUDA backend for 03 · Reverse Array. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def reverse_(x):
    if not IMPLEMENTED:
        raise NotImplementedError("reverse_: CUDA backend not implemented (cuda/kernels.cu)")
    load_extension("gpu_lab_reverse_array", Path(__file__).parent).reverse_(x)
    return x
