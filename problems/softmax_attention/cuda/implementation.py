"""CUDA backend for 40 · Softmax Attention. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def attention(q, k, v):
    if not IMPLEMENTED:
        raise NotImplementedError("attention: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(q)
    load_extension("gpu_lab_softmax_attention", Path(__file__).parent).attention(q, k, v, out)
    return out
