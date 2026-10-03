"""Test and benchmark cases for 38 · Rotary Positional Embedding.

Shared by the tests and by `python -m lab.bench`.
"""

import torch

from lab.data import element_size

TEST_CASES = [{"b": 1, "s": 1, "h": 1, "d": 2}, {"b": 2, "s": 5, "h": 3, "d": 8}]
BENCH_CASES = [{"b": 8, "s": 4096, "h": 32, "d": 128}, {"b": 1, "s": 32768, "h": 8, "d": 128}]


def make_inputs(c, s):
    """Positional arguments for api.rope, drawn from the lab.data.Sampler `s`."""
    theta = s.uniform(0, 6.283, c["s"], c["d"] // 2, dtype=torch.float32)
    x = s.randn(c["b"], c["s"], c["h"], c["d"])
    return (x, theta.cos().to(s.dtype), theta.sin().to(s.dtype))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return (2 * c["b"] * c["s"] * c["h"] * c["d"] + c["s"] * c["d"]) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
