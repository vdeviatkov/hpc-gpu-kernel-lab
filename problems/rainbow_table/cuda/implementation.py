"""CUDA hash variants, built lazily and launched on the current PyTorch stream."""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

def rainbow_table(x, rounds, *, variant=1, threads=256):
    out = torch.empty(x.shape, dtype=torch.uint32, device=x.device)
    load_extension("gpu_lab_rainbow_table", Path(__file__).parent).rainbow_table(
        x, rounds, out, variant, threads
    )
    return out
