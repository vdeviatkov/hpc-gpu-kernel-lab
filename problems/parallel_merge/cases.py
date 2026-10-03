"""Test and benchmark cases for 20 · Parallel Merge.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"m": 1, "n": 1}, {"m": 3, "n": 5}, {"m": 1000, "n": 1025}]
BENCH_CASES = [{"m": 12500000, "n": 12500000}, {"m": 1000000, "n": 24000000}]


def make_inputs(c, s):
    """Positional arguments for api.merge, drawn from the lab.data.Sampler `s`."""
    return (s.sorted(c["m"]), s.sorted(c["n"]))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 2 * (c["m"] + c["n"]) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
