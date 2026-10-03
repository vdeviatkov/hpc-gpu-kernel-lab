"""CUDA backend for 29 · Fused GEMM + Bias + Activation. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def linear_relu(x, w, bias):
    if not IMPLEMENTED:
        raise NotImplementedError("linear_relu: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((x.shape[0], w.shape[1]), dtype=x.dtype, device=x.device)
    load_extension("gpu_lab_fused_gemm_epilogue", Path(__file__).parent).linear_relu(
        x, w, bias, out
    )
    return out
