"""CUDA backend for 49 · Paged Attention. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def paged_attention(q, k_cache, v_cache, block_table, context_lens):
    if not IMPLEMENTED:
        raise NotImplementedError("paged_attention: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(q)
    load_extension("gpu_lab_paged_attention", Path(__file__).parent).paged_attention(
        q, k_cache, v_cache, block_table, context_lens, out
    )
    return out
