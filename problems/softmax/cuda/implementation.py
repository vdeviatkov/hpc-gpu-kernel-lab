"""CUDA backend for 32 · Softmax. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def softmax(x):
    if not IMPLEMENTED:
        raise NotImplementedError("softmax: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(x)
    load_extension("gpu_lab_softmax", Path(__file__).parent).softmax(x, out)
    return out
