"""CUDA backend for 39 · SwiGLU MLP Block. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def swiglu_mlp(x, w_gate, w_up, w_down):
    if not IMPLEMENTED:
        raise NotImplementedError("swiglu_mlp: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(x)
    load_extension("gpu_lab_swiglu_mlp", Path(__file__).parent).swiglu_mlp(
        x, w_gate, w_up, w_down, out
    )
    return out
