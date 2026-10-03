"""Test and benchmark cases for 30 · Sigmoid Linear Unit.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"n": 1}, {"n": 3}, {"n": 1025}, {"n": 65537}]
BENCH_CASES = [{"n": 1048576}, {"n": 25000000}]


def make_inputs(c, s):
    """Positional arguments for api.silu, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["n"], scale=4.0),)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 2 * c["n"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
