"""Test and benchmark cases for 40 · Softmax Attention.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"m": 1, "n": 1, "d": 1}, {"m": 3, "n": 5, "d": 8}, {"m": 65, "n": 129, "d": 64}]
BENCH_CASES = [{"m": 4096, "n": 4096, "d": 64}, {"m": 1024, "n": 16384, "d": 128}]


def make_inputs(c, s):
    """Positional arguments for api.attention, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["m"], c["d"]), s.randn(c["n"], c["d"]), s.randn(c["n"], c["d"]))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return (2 * c["m"] * c["d"] + 2 * c["n"] * c["d"]) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 4 * c["m"] * c["n"] * c["d"]
