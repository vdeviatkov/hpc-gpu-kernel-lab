"""PyTorch reference and eager baselines for 05 · RGB to Grayscale.

method="formula" is the reference every other backend is tested against: the statement's
formula with its exact coefficients, computed in FP32 and rounded to the image dtype. It
runs as several separate kernels (one per multiply and add), each a full pass over memory.

Two faster baselines show what PyTorch itself can reach:
- method="matmul": `image @ w`, one cuBLAS matrix-vector product (inner dimension 3);
- method="compile": torch.compile of the formula, which fuses it into one generated kernel.
  It compiles one static kernel per image shape and dtype. dynamic=True would compile a single
  kernel for every size, but Inductor sizes its blocks from the first image it sees and keeps
  that choice in its on-disk cache: after a 2 x 2 image, 4096 x 4096 FP32 ran 6x slower.
  Static compilation recompiles per shape. The first call for each shape raises the recompile
  limit, so reaching it raises an error instead of silently falling back to eager execution;
  later calls skip that step, which would add about 12 µs per call.
"""

from functools import cache

import torch

COEFFICIENTS = (0.299, 0.587, 0.114)
# Distinct shapes and dtypes this baseline may compile in one process (the tests use about 80).
RECOMPILE_LIMIT = 256


def _formula(image):
    rgb = image.float()
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.to(image.dtype)


@cache
def _compiled():
    return torch.compile(_formula, dynamic=False)


_compiled_keys = set()


def _run_compiled(image):
    key = (image.shape, image.dtype, image.device)
    if key in _compiled_keys:
        return _compiled()(image)
    with torch._dynamo.config.patch(
        recompile_limit=RECOMPILE_LIMIT, fail_on_recompile_limit_hit=True
    ):
        out = _compiled()(image)
    _compiled_keys.add(key)
    return out


def rgb_to_grayscale(image, *, method="formula"):
    if method == "matmul":
        weights = torch.tensor(COEFFICIENTS, dtype=image.dtype, device=image.device)
        return image @ weights
    if method == "compile":
        return _run_compiled(image)
    return _formula(image)
