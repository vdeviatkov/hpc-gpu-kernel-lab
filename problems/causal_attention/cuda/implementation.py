"""CUDA backend for 41 · Causal Self-Attention. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def causal_attention(q, k, v):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "causal_attention: CUDA backend not implemented (cuda/kernels.cu)"
        )
    out = torch.empty_like(q)
    load_extension("gpu_lab_causal_attention", Path(__file__).parent).causal_attention(q, k, v, out)
    return out
