"""CUDA backend for 23 · Matrix Multiplication. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def matmul(a, b):
    if not IMPLEMENTED:
        raise NotImplementedError("matmul: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    load_extension("gpu_lab_matrix_multiplication_fp32", Path(__file__).parent).matmul(a, b, out)
    return out
