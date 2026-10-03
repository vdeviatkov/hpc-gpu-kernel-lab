"""CUDA backend for 02 · ReLU. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def relu(x):
    if not IMPLEMENTED:
        raise NotImplementedError("relu: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(x)
    load_extension("gpu_lab_relu", Path(__file__).parent).relu(x, out)
    return out
