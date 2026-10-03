"""Test and benchmark cases for 31 · Swish-Gated Linear Unit.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"m": 1, "d": 1}, {"m": 3, "d": 5}, {"m": 65, "d": 129}]
BENCH_CASES = [{"m": 4096, "d": 4096}, {"m": 16384, "d": 11008}]


def make_inputs(c, s):
    """Positional arguments for api.swiglu, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["m"], 2 * c["d"]),)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 3 * c["m"] * c["d"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
