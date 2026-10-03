"""CUDA backend for 06 · Rainbow Table. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def rainbow_table(x, rounds):
    if not IMPLEMENTED:
        raise NotImplementedError("rainbow_table: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(x)
    load_extension("gpu_lab_rainbow_table", Path(__file__).parent).rainbow_table(x, rounds, out)
    return out
