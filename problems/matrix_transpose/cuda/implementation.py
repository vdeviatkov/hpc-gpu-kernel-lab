"""CUDA transpose variants, built lazily and launched on the current PyTorch stream."""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension


def extension():
    return load_extension("gpu_lab_matrix_transpose", Path(__file__).parent)


def transpose(a, *, variant=3):
    out = torch.empty((a.shape[1], a.shape[0]), dtype=a.dtype, device=a.device)
    extension().transpose(a, out, variant)
    return out
