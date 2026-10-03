"""Test and benchmark cases for 35 · Batch Normalization.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"n": 1, "c": 1, "h": 1, "w": 2}, {"n": 3, "c": 5, "h": 7, "w": 9}]
BENCH_CASES = [{"n": 32, "c": 64, "h": 56, "w": 56}, {"n": 8, "c": 256, "h": 28, "w": 28}]


def make_inputs(c, s):
    """Positional arguments for api.batch_norm, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["n"], c["c"], c["h"], c["w"]), s.randn(c["c"]), s.randn(c["c"]), 1e-5)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 2 * c["n"] * c["c"] * c["h"] * c["w"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
