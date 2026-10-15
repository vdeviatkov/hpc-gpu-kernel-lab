"""CUDA backend for 05 · RGB to Grayscale. Builds lazily on the first call and launches on
PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension


def rgb_to_grayscale(image, *, variant=0):
    out = torch.empty(image.shape[:2], dtype=image.dtype, device=image.device)
    extension = load_extension("gpu_lab_rgb_to_grayscale", Path(__file__).parent)
    extension.rgb_to_grayscale_out(image, out, variant)
    return out
