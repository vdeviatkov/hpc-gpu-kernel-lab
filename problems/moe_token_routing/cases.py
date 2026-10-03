"""Test and benchmark cases for 50 · MoE Token Routing and Dispatch.

Shared by the tests and by `python -m lab.bench`.
"""

import torch

from lab.data import element_size

TEST_CASES = [{"t": 1, "d": 1, "e": 1, "k": 1}, {"t": 17, "d": 8, "e": 8, "k": 2}]
BENCH_CASES = [{"t": 16384, "d": 4096, "e": 64, "k": 2}, {"t": 4096, "d": 7168, "e": 256, "k": 8}]


def make_inputs(c, s):
    """Positional arguments for api.route_tokens, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["t"], c["d"]), s.randn(c["t"], c["e"], dtype=torch.float32), c["k"])


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return (c["t"] * c["d"] + c["t"] * c["k"] * c["d"]) * e + c["t"] * c["e"] * 4


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
