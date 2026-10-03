"""CUDA backend for 43 · Attention with Linear Biases. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def alibi_attention(q, k, v, slopes):
    if not IMPLEMENTED:
        raise NotImplementedError("alibi_attention: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(q)
    load_extension("gpu_lab_alibi_attention", Path(__file__).parent).alibi_attention(
        q, k, v, slopes, out
    )
    return out
