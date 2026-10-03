"""Build once, lazily; all device launches use PyTorch's current stream."""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension


def add(a, b, *, out=None, variant=0, threads=256):
    if out is None:
        out = torch.empty_like(a)
    load_extension("gpu_lab_vector_add", Path(__file__).parent).add_out(a, b, out, variant, threads)
    return out
