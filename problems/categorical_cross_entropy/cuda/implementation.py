"""CUDA backend for 37 · Categorical Cross Entropy Loss. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def cross_entropy(logits, labels):
    if not IMPLEMENTED:
        raise NotImplementedError("cross_entropy: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((), dtype=torch.float32, device=logits.device)
    load_extension("gpu_lab_categorical_cross_entropy", Path(__file__).parent).cross_entropy(
        logits, labels, out
    )
    return out
