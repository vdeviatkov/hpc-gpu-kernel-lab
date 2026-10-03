"""CUDA backend for 26 · FP16 Batched Matrix Multiplication. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def bmm(a, b):
    if not IMPLEMENTED:
        raise NotImplementedError("bmm: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=a.dtype, device=a.device)
    load_extension("gpu_lab_batched_matmul_fp16", Path(__file__).parent).bmm(a, b, out)
    return out
