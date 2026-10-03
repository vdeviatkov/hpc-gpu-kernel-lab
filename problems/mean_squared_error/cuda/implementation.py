"""CUDA backend for 15 · Mean Squared Error. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def mse(pred, target):
    if not IMPLEMENTED:
        raise NotImplementedError("mse: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((), dtype=torch.float32, device=pred.device)
    load_extension("gpu_lab_mean_squared_error", Path(__file__).parent).mse(pred, target, out)
    return out
