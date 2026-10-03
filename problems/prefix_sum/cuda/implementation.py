"""CUDA backend for 17 · Prefix Sum. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def inclusive_scan(x):
    if not IMPLEMENTED:
        raise NotImplementedError("inclusive_scan: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(x)
    load_extension("gpu_lab_prefix_sum", Path(__file__).parent).inclusive_scan(x, out)
    return out
