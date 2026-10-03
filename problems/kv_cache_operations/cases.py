"""Test and benchmark cases for 47 · KV-Cache Update and Paged Access.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [
    {"b": 1, "h": 1, "t": 1, "d": 1, "block_size": 1, "max_blocks": 1},
    {"b": 3, "h": 2, "t": 5, "d": 8, "block_size": 4, "max_blocks": 4},
]
BENCH_CASES = [
    {"b": 64, "h": 32, "t": 1, "d": 128, "block_size": 16, "max_blocks": 256},
    {"b": 8, "h": 32, "t": 512, "d": 128, "block_size": 16, "max_blocks": 64},
]


def make_inputs(c, s):
    """Positional arguments for api.kv_cache_append, drawn from the lab.data.Sampler `s`."""
    b, h, t, d, bs, mb = c["b"], c["h"], c["t"], c["d"], c["block_size"], c["max_blocks"]
    table = s.permutation(b * mb).reshape(b, mb)
    positions = s.randint(0, mb * bs - t + 1, b)
    return (
        s.randn(b, h, t, d),
        s.randn(b, h, t, d),
        s.randn(b * mb, h, bs, d),
        s.randn(b * mb, h, bs, d),
        table,
        positions,
    )


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 4 * c["b"] * c["h"] * c["t"] * c["d"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
