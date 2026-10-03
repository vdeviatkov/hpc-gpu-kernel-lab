"""CUDA backend for 18 · Histogramming. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def histogram(x, bins):
    if not IMPLEMENTED:
        raise NotImplementedError("histogram: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty(bins, dtype=torch.int32, device=x.device)
    load_extension("gpu_lab_histogramming", Path(__file__).parent).histogram(x, bins, out)
    return out
