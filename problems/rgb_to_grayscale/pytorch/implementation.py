"""PyTorch reference and eager baselines for 05 · RGB to Grayscale.

method="formula" is the reference every other backend is tested against: the statement's
formula with its exact coefficients, computed in FP32 and rounded to the image dtype. It
runs as several separate kernels (one per multiply and add), each a full pass over memory.

Two faster baselines show what PyTorch itself can reach:
- method="matmul": `image @ w`, one cuBLAS matrix-vector product (inner dimension 3);
- method="compile": torch.compile of the formula, which fuses it into one generated kernel,
  compiled once per image shape and dtype (lab.compile explains why not dynamic=True).
"""

import torch

from lab.compile import StaticCompiled

COEFFICIENTS = (0.299, 0.587, 0.114)


def _formula(image):
    rgb = image.float()
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.to(image.dtype)


_compiled = StaticCompiled(_formula)


def rgb_to_grayscale(image, *, method="formula"):
    if method == "matmul":
        weights = torch.tensor(COEFFICIENTS, dtype=image.dtype, device=image.device)
        return image @ weights
    if method == "compile":
        return _compiled(image)
    return _formula(image)
