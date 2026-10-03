"""Test and benchmark cases for 32 · Softmax.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"m": 1, "n": 1}, {"m": 3, "n": 5}, {"m": 65, "n": 1025}]
BENCH_CASES = [{"m": 4096, "n": 4096}, {"m": 1, "n": 1048576}]


def make_inputs(c, s):
    """Positional arguments for api.softmax, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["m"], c["n"], scale=4.0),)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 2 * c["m"] * c["n"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
