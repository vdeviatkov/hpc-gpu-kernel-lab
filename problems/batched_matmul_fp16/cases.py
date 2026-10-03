"""Test and benchmark cases for 26 · FP16 Batched Matrix Multiplication.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"batch": 1, "m": 1, "k": 1, "n": 1}, {"batch": 3, "m": 5, "k": 7, "n": 9}]
BENCH_CASES = [
    {"batch": 32, "m": 1024, "k": 1024, "n": 1024},
    {"batch": 256, "m": 128, "k": 128, "n": 128},
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
