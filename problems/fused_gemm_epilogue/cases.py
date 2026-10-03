"""Test and benchmark cases for 29 · Fused GEMM + Bias + Activation.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"m": 1, "k": 1, "n": 1}, {"m": 3, "k": 5, "n": 7}, {"m": 65, "k": 33, "n": 129}]
BENCH_CASES = [{"m": 4096, "k": 4096, "n": 4096}, {"m": 8192, "k": 1024, "n": 4096}]


def make_inputs(c, s):
    """Positional arguments for api.linear_relu, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["m"], c["k"]), s.randn(c["k"], c["n"], scale=c["k"] ** -0.5), s.randn(c["n"]))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return (c["m"] * c["k"] + c["k"] * c["n"] + c["n"] + c["m"] * c["n"]) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 2 * c["m"] * c["n"] * c["k"]
