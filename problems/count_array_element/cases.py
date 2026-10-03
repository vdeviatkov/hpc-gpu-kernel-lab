"""Test and benchmark cases for 14 · Count Array Element.

Shared by the tests and by `python -m lab.bench`.
"""

TEST_CASES = [{"n": 1}, {"n": 3}, {"n": 1025}, {"n": 65537}]
BENCH_CASES = [{"n": 1048576}, {"n": 25000000}]


def make_inputs(c, s):
    """Positional arguments for api.count_equal, drawn from the lab.data.Sampler `s`."""
    return (s.randint(0, 10, c["n"]), 3)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    del dtype  # Byte count is dtype-independent.
    return c["n"] * 4


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
