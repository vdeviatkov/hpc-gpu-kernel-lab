"""CUDA backend for 42 · Multi-Head Attention. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def mha(q, k, v):
    if not IMPLEMENTED:
        raise NotImplementedError("mha: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(q)
    load_extension("gpu_lab_multi_head_attention", Path(__file__).parent).mha(q, k, v, out)
    return out
