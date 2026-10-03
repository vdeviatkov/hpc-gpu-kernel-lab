"""Test and benchmark cases for 25 · General Matrix Multiplication (GEMM) — FP16.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"m": 1, "k": 1, "n": 1}, {"m": 3, "k": 5, "n": 7}, {"m": 65, "k": 33, "n": 129}]
BENCH_CASES = [{"m": 1024, "k": 1024, "n": 1024}, {"m": 8192, "k": 8192, "n": 8192}]


def make_inputs(c, s):
    """Positional arguments for api.gemm, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["m"], c["k"]), s.randn(c["k"], c["n"]), s.randn(c["m"], c["n"]), 1.0, 0.5)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return (c["m"] * c["k"] + c["k"] * c["n"] + 2 * c["m"] * c["n"]) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 2 * c["m"] * c["n"] * c["k"]
