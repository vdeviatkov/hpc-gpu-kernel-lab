"""Test and benchmark cases for 07 · Matrix Copy.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"m": 1, "n": 1}, {"m": 3, "n": 5}, {"m": 65, "n": 129}]
BENCH_CASES = [{"m": 1024, "n": 1024}, {"m": 8192, "n": 8192}]


def make_inputs(c, s):
    """Positional arguments for api.matrix_copy, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["m"], c["n"]),)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 2 * c["m"] * c["n"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
