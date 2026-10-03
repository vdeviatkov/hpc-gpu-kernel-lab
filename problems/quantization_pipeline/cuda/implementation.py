"""CUDA backend for 46 · Quantize / Dequantize Pipeline. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def quantize(x, group_size):
    if not IMPLEMENTED:
        raise NotImplementedError("quantize: CUDA backend not implemented (cuda/kernels.cu)")
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty(
        (x.shape[0], x.shape[1] // group_size), dtype=torch.float32, device=x.device
    )
    load_extension("gpu_lab_quantization_pipeline", Path(__file__).parent).quantize(
        x, group_size, q, scale
    )
    return (q, scale)
