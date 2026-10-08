"""CUDA backend for 03 · Reverse Array. Builds lazily on the first call and launches on
PyTorch's current stream.
"""

from pathlib import Path

from lab.cuda_ext import load_extension


def reverse_(x, *, variant=0, threads=256):
    load_extension("gpu_lab_reverse_array", Path(__file__).parent).reverse_(x, variant, threads)
    return x
