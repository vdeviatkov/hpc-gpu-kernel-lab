"""Test and benchmark cases for 24 · Batched Matrix Multiplication — FP32.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"batch": 1, "m": 1, "k": 1, "n": 1}, {"batch": 3, "m": 5, "k": 7, "n": 9}]
BENCH_CASES = [
    {"batch": 32, "m": 512, "k": 512, "n": 512},
    {"batch": 128, "m": 256, "k": 256, "n": 256},
]


def make_inputs(c, s):
    """Positional arguments for api.bmm, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["batch"], c["m"], c["k"]), s.randn(c["batch"], c["k"], c["n"]))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return c["batch"] * (c["m"] * c["k"] + c["k"] * c["n"] + c["m"] * c["n"]) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 2 * c["batch"] * c["m"] * c["n"] * c["k"]
