"""Test and benchmark cases for 45 · Decaying Causal Attention.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"s": 1, "d": 1}, {"s": 5, "d": 8}, {"s": 129, "d": 64}]
BENCH_CASES = [{"s": 4096, "d": 64}, {"s": 16384, "d": 128}]


def make_inputs(c, s):
    """Positional arguments for api.decaying_attention, drawn from the lab.data.Sampler `s`."""
    return (*(s.randn(c["s"], c["d"]) for _ in range(3)), 0.95)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 4 * c["s"] * c["d"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 2 * c["s"] * c["s"] * c["d"]
