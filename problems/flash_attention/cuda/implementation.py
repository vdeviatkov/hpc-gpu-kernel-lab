"""CUDA backend for 48 · FlashAttention-Style Online Attention. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def flash_attention(q, k, v, causal):
    if not IMPLEMENTED:
        raise NotImplementedError("flash_attention: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(q)
    load_extension("gpu_lab_flash_attention", Path(__file__).parent).flash_attention(
        q, k, v, causal, out
    )
    return out
