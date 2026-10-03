"""CUDA backend for 27 · INT8 Quantized MatMul. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def int8_matmul(a, b, scale_a, scale_b, zero_a, zero_b):
    if not IMPLEMENTED:
        raise NotImplementedError("int8_matmul: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device=a.device)
    load_extension("gpu_lab_int8_quantized_matmul", Path(__file__).parent).int8_matmul(
        a, b, scale_a, scale_b, zero_a, zero_b, out
    )
    return out
