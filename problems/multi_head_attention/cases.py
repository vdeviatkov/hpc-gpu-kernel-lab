"""Test and benchmark cases for 42 · Multi-Head Attention.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"b": 1, "h": 1, "s": 1, "d": 1}, {"b": 2, "h": 3, "s": 17, "d": 16}]
BENCH_CASES = [{"b": 8, "h": 16, "s": 2048, "d": 64}, {"b": 1, "h": 32, "s": 8192, "d": 128}]


def make_inputs(c, s):
    """Positional arguments for api.mha, drawn from the lab.data.Sampler `s`."""
    return tuple(s.randn(c["b"], c["h"], c["s"], c["d"]) for _ in range(3))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 4 * c["b"] * c["h"] * c["s"] * c["d"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 4 * c["b"] * c["h"] * c["s"] ** 2 * c["d"]
