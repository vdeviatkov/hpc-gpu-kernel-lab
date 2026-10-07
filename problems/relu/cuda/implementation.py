"""CUDA backend for 02 · ReLU. Builds lazily on the first call and launches on
PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension


def relu(x, *, variant=0, threads=256):
    out = torch.empty_like(x)
    load_extension("gpu_lab_relu", Path(__file__).parent).relu_out(x, out, variant, threads)
    return out
