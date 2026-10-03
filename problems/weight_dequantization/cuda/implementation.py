"""CUDA backend for 12 · Weight Dequantization. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def dequantize(q, scale, quant_block):
    if not IMPLEMENTED:
        raise NotImplementedError("dequantize: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty(q.shape, dtype=scale.dtype, device=q.device)
    load_extension("gpu_lab_weight_dequantization", Path(__file__).parent).dequantize(
        q, scale, quant_block, out
    )
    return out
