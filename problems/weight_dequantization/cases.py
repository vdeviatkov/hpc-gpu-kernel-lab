"""Test and benchmark cases for 12 · Weight Dequantization.

Shared by the tests and by `python -m lab.bench`.
"""

import torch

from lab.data import element_size

TEST_CASES = [
    {"m": 1, "n": 1, "block": 1},
    {"m": 33, "n": 65, "block": 16},
    {"m": 128, "n": 256, "block": 128},
]
BENCH_CASES = [{"m": 4096, "n": 4096, "block": 128}, {"m": 8192, "n": 8192, "block": 128}]


def make_inputs(c, s):
    """Positional arguments for api.dequantize, drawn from the lab.data.Sampler `s`."""
    b = c["block"]
    scale = s.uniform(1e-3, 0.1, -(-c["m"] // b), -(-c["n"] // b))
    return (s.randint(-128, 128, c["m"], c["n"], dtype=torch.int8), scale, b)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return c["m"] * c["n"] * (1 + e)


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
