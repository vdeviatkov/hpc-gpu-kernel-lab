"""CUDA backend for 25 · General Matrix Multiplication (GEMM) — FP16. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def gemm(a, b, c, alpha, beta):
    if not IMPLEMENTED:
        raise NotImplementedError("gemm: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(c)
    load_extension("gpu_lab_gemm_fp16", Path(__file__).parent).gemm(a, b, c, alpha, beta, out)
    return out
