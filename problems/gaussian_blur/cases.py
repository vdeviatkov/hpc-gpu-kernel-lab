"""Test and benchmark cases for 10 · Gaussian Blur.

Shared by the tests and by `python -m lab.bench`.
"""

import torch

from lab.data import element_size

TEST_CASES = [{"h": 1, "w": 1, "k": 1}, {"h": 9, "w": 13, "k": 3}, {"h": 64, "w": 67, "k": 5}]
BENCH_CASES = [{"h": 2048, "w": 2048, "k": 5}, {"h": 4096, "w": 4096, "k": 15}]


def make_inputs(c, s):
    """Positional arguments for api.gaussian_blur, drawn from the lab.data.Sampler `s`."""
    kernel = s.uniform(0.1, 1, c["k"], c["k"], dtype=torch.float32)
    return (s.randn(c["h"], c["w"]), (kernel / kernel.sum()).to(s.dtype))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return (2 * c["h"] * c["w"] + c["k"] ** 2) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 2 * c["h"] * c["w"] * c["k"] ** 2
