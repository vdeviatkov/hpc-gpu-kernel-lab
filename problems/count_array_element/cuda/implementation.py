"""CUDA backend for 14 · Count Array Element. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def count_equal(x, value):
    if not IMPLEMENTED:
        raise NotImplementedError("count_equal: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty((), dtype=torch.int32, device=x.device)
    load_extension("gpu_lab_count_array_element", Path(__file__).parent).count_equal(x, value, out)
    return out
