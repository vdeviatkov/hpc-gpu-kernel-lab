"""Test and benchmark cases for 21 · Dense GEMV.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"m": 1, "k": 1}, {"m": 3, "k": 5}, {"m": 65, "k": 257}]
BENCH_CASES = [{"m": 4096, "k": 4096}, {"m": 16384, "k": 16384}]


def make_inputs(c, s):
    """Positional arguments for api.gemv, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["m"], c["k"]), s.randn(c["k"]))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return (c["m"] * c["k"] + c["k"] + c["m"]) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 2 * c["m"] * c["k"]
