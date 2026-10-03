"""Test and benchmark cases for 17 · Prefix Sum.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"n": 1}, {"n": 2}, {"n": 1025}, {"n": 65537}]
BENCH_CASES = [{"n": 1048576}, {"n": 25000000}]


def make_inputs(c, s):
    """Positional arguments for api.inclusive_scan, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["n"]),)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 2 * c["n"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return c["n"]
