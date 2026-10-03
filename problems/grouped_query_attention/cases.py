"""Test and benchmark cases for 44 · Grouped Query Attention.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [
    {"b": 1, "hq": 1, "hkv": 1, "s": 1, "d": 1},
    {"b": 2, "hq": 8, "hkv": 2, "s": 17, "d": 16},
]
BENCH_CASES = [
    {"b": 8, "hq": 32, "hkv": 8, "s": 2048, "d": 128},
    {"b": 1, "hq": 64, "hkv": 8, "s": 8192, "d": 128},
]


def make_inputs(c, s):
    """Positional arguments for api.gqa, drawn from the lab.data.Sampler `s`."""
    q = s.randn(c["b"], c["hq"], c["s"], c["d"])
    return (q, s.randn(c["b"], c["hkv"], c["s"], c["d"]), s.randn(c["b"], c["hkv"], c["s"], c["d"]))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 2 * c["b"] * c["s"] * c["d"] * (c["hq"] + c["hkv"]) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 2 * c["b"] * c["hq"] * c["s"] ** 2 * c["d"]
