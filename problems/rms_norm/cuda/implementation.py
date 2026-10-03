"""CUDA backend for 33 · RMS Normalization. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def rms_norm(x, weight, eps):
    if not IMPLEMENTED:
        raise NotImplementedError("rms_norm: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(x)
    load_extension("gpu_lab_rms_norm", Path(__file__).parent).rms_norm(x, weight, eps, out)
    return out
