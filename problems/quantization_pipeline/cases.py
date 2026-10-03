"""Test and benchmark cases for 46 · Quantize / Dequantize Pipeline.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [
    {"m": 1, "n": 1, "group": 1},
    {"m": 3, "n": 64, "group": 32},
    {"m": 65, "n": 1024, "group": 128},
]
BENCH_CASES = [{"m": 4096, "n": 4096, "group": 128}, {"m": 16384, "n": 8192, "group": 64}]


def make_inputs(c, s):
    """Positional arguments for api.quantize, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["m"], c["n"]), c["group"])


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return c["m"] * c["n"] * (e + 1) + 4 * c["m"] * c["n"] // c["group"]


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
