"""PyTorch reference and eager baselines for 05 · RGB to Grayscale.

method="formula" is the reference every other backend is tested against: the statement's
formula with its exact coefficients, computed in FP32 and rounded to the image dtype. It
runs as several separate kernels (one per multiply and add), each a full pass over memory.

Two faster baselines show what PyTorch itself can reach:
- method="matmul": `image @ w`, one cuBLAS matrix-vector product (inner dimension 3);
- method="compile": torch.compile of the formula, which fuses it into one generated kernel.
  dynamic=True compiles one kernel for every image size; static compilation would recompile
  per shape and, past the recompile limit, silently fall back to eager execution.
"""

from functools import cache

import torch

COEFFICIENTS = (0.299, 0.587, 0.114)


def _formula(image):
    rgb = image.float()
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.to(image.dtype)


@cache
def _compiled():
    return torch.compile(_formula, dynamic=True)


def rgb_to_grayscale(image, *, method="formula"):
    if method == "matmul":
        weights = torch.tensor(COEFFICIENTS, dtype=image.dtype, device=image.device)
        return image @ weights
    if method == "compile":
        return _compiled()(image)
    return _formula(image)
