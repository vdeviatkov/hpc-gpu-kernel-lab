"""CUDA backend for 08 · Matrix Transpose. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def transpose(a):
    if not IMPLEMENTED:
        raise NotImplementedError("transpose: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((a.shape[1], a.shape[0]), dtype=a.dtype, device=a.device)
    load_extension("gpu_lab_matrix_transpose", Path(__file__).parent).transpose(a, out)
    return out
