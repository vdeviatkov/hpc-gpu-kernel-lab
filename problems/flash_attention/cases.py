"""Test and benchmark cases for 48 · FlashAttention-Style Online Attention.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [
    {"b": 1, "h": 1, "s": 1, "d": 16, "causal": False},
    {"b": 2, "h": 3, "s": 65, "d": 64, "causal": True},
]
BENCH_CASES = [
    {"b": 8, "h": 16, "s": 2048, "d": 64, "causal": True},
    {"b": 1, "h": 32, "s": 16384, "d": 128, "causal": True},
]


def make_inputs(c, s):
    """Positional arguments for api.flash_attention, drawn from the lab.data.Sampler `s`."""
    return (*(s.randn(c["b"], c["h"], c["s"], c["d"]) for _ in range(3)), c["causal"])


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 4 * c["b"] * c["h"] * c["s"] * c["d"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return (2 if c["causal"] else 4) * c["b"] * c["h"] * c["s"] ** 2 * c["d"]
