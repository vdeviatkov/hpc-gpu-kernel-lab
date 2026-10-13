"""CUDA backend for 04 · Interleave Arrays. Builds lazily on the first call and launches on
PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension


def interleave(a, b, *, variant=0, threads=256):
    out = torch.empty(2 * a.numel(), dtype=a.dtype, device=a.device)
    extension = load_extension("gpu_lab_interleave_arrays", Path(__file__).parent)
    extension.interleave_out(a, b, out, variant, threads)
    return out
