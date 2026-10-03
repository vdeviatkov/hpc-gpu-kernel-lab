"""Test and benchmark cases for 18 · Histogramming.

Shared by the tests and by `python -m lab.bench`.
"""

TEST_CASES = [{"n": 1, "bins": 1}, {"n": 1025, "bins": 16}, {"n": 65537, "bins": 256}]
BENCH_CASES = [{"n": 25000000, "bins": 256}, {"n": 25000000, "bins": 4096}]


def make_inputs(c, s):
    """Positional arguments for api.histogram, drawn from the lab.data.Sampler `s`."""
    return (s.randint(0, c["bins"], c["n"]), c["bins"])


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    del dtype  # Byte count is dtype-independent.
    return c["n"] * 4 + c["bins"] * 4


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
