"""CUDA backend for 10 · Gaussian Blur. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def gaussian_blur(image, kernel):
    if not IMPLEMENTED:
        raise NotImplementedError("gaussian_blur: CUDA backend not implemented (cuda/kernels.cu)")
    out = torch.empty_like(image)
    load_extension("gpu_lab_gaussian_blur", Path(__file__).parent).gaussian_blur(image, kernel, out)
    return out
