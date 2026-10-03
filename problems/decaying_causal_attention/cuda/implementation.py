"""CUDA backend for 45 · Decaying Causal Attention. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def decaying_attention(q, k, v, gamma):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "decaying_attention: CUDA backend not implemented (cuda/kernels.cu)"
        )
    out = torch.empty_like(q)
    load_extension("gpu_lab_decaying_causal_attention", Path(__file__).parent).decaying_attention(
        q, k, v, gamma, out
    )
    return out
