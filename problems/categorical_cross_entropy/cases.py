"""Test and benchmark cases for 37 · Categorical Cross Entropy Loss.

Shared by the tests and by `python -m lab.bench`.
"""

import torch

from lab.data import element_size

TEST_CASES = [{"m": 1, "c": 1}, {"m": 3, "c": 5}, {"m": 65, "c": 1025}]
BENCH_CASES = [{"m": 4096, "c": 32000}, {"m": 65536, "c": 1000}]


def make_inputs(c, s):
    """Positional arguments for api.cross_entropy, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["m"], c["c"], scale=3.0), s.randint(0, c["c"], c["m"], dtype=torch.int64))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return c["m"] * c["c"] * e + c["m"] * 8


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
