"""CUDA copy variants, built lazily and launched on the current PyTorch stream."""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension


def extension():
    return load_extension("gpu_lab_matrix_copy", Path(__file__).parent)


def matrix_copy(a, *, variant=2, threads=256):
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    extension().matrix_copy(a, out, variant, threads)
    return out
