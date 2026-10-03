"""Test and benchmark cases for 09 · 1D Convolution.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"n": 1, "k": 1}, {"n": 17, "k": 3}, {"n": 1025, "k": 31}]
BENCH_CASES = [{"n": 1048576, "k": 31}, {"n": 25000000, "k": 255}]


def make_inputs(c, s):
    """Positional arguments for api.conv1d, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["n"]), s.randn(c["k"]))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return (2 * c["n"] - c["k"] + 1 + c["k"]) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 2 * (c["n"] - c["k"] + 1) * c["k"]
