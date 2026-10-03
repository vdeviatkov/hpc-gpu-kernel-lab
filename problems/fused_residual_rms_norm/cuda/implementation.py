"""CUDA backend for 36 · Fused Residual Add and RMS Norm. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def fused_add_rms_norm(x, residual, weight, eps):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "fused_add_rms_norm: CUDA backend not implemented (cuda/kernels.cu)"
        )
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    load_extension("gpu_lab_fused_residual_rms_norm", Path(__file__).parent).fused_add_rms_norm(
        x, residual, weight, eps, out, residual_out
    )
    return (out, residual_out)
