"""06 · Rainbow Table: public entry point.

Contract, verified against the LeetGPU statement (challenges/easy/24_rainbow_table):

    fnv1a(x):  h = 2166136261
               for byte k = 0, 1, 2, 3 of x (least significant first):
                   h = (h ^ byte_k) * 16777619      (mod 2^32)
    out[i] = fnv1a applied R times to input[i]   (each round hashes the previous result)

- Source: int32 input of N values in [0, 2^31 - 1], uint32 output; 1 <= N <= 10,000,000,
  1 <= R <= 100. The performance test is N = 5,000,000, R = 10. Results must match
  exactly (atol = rtol = 0).
- Here: a contiguous 1-D int32 tensor and a Python int `rounds` in, a new uint32 tensor out.
- Lab extensions: N = 0; R = 0 (the input reinterpreted as uint32) and R > 100; negative
  inputs (the hash reads only the 32-bit pattern, so they behave like their uint32 value).
"""

import operator
import sys

import torch

from lab import dispatch

FUNCTION = "rainbow_table"
VARIANTS = {
    # torch.compile of the hash in 32-bit integers: all R rounds fused into one kernel.
    "pytorch_compile": ("pytorch", {"method": "compile"}),
}
BACKENDS = ("pytorch", *VARIANTS, "cuda")
# The input dtype; the output is always uint32.
DTYPES = {
    "int32": torch.int32,
}
# Integer hashing: every backend must match the reference bit for bit.
TOLERANCES = {
    torch.int32: (0, 0),
}


def rainbow_table(x, rounds, *, backend="pytorch", **options):
    """Return a new uint32 tensor: FNV-1a applied `rounds` times to every element of x."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected torch.Tensor")
    if x.layout != torch.strided or x.ndim != 1 or not x.is_contiguous():
        raise ValueError("expected a contiguous one-dimensional tensor")
    if x.dtype != torch.int32:
        raise ValueError("supported input dtype: int32")
    if isinstance(rounds, bool):
        raise TypeError("rounds must be an integer")
    try:
        rounds = operator.index(rounds)
    except TypeError:
        raise TypeError("rounds must be an integer") from None
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, rounds, **options)
