"""Test and benchmark cases for 49 · Paged Attention.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [
    {"b": 1, "h": 1, "d": 8, "block_size": 1, "max_blocks": 1},
    {"b": 3, "h": 2, "d": 16, "block_size": 4, "max_blocks": 8},
]
BENCH_CASES = [
    {"b": 64, "h": 32, "d": 128, "block_size": 16, "max_blocks": 128},
    {"b": 8, "h": 32, "d": 128, "block_size": 16, "max_blocks": 2048},
]


def make_inputs(c, s):
    """Positional arguments for api.paged_attention, drawn from the lab.data.Sampler `s`."""
    b, h, d, bs, mb = c["b"], c["h"], c["d"], c["block_size"], c["max_blocks"]
    table = s.permutation(b * mb).reshape(b, mb)
    return (
        s.randn(b, h, d),
        s.randn(b * mb, h, bs, d),
        s.randn(b * mb, h, bs, d),
        table,
        s.randint(1, mb * bs + 1, b),
    )


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return c["b"] * c["h"] * c["max_blocks"] * c["block_size"] * c["d"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
