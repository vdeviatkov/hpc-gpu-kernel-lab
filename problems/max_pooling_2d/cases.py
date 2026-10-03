"""Test and benchmark cases for 11 · 2D Max Pooling.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [
    {"n": 1, "c": 1, "h": 4, "w": 4, "kernel": 2, "stride": 2, "padding": 0},
    {"n": 2, "c": 3, "h": 9, "w": 11, "kernel": 3, "stride": 2, "padding": 1},
]
BENCH_CASES = [
    {"n": 32, "c": 64, "h": 112, "w": 112, "kernel": 3, "stride": 2, "padding": 1},
    {"n": 8, "c": 256, "h": 56, "w": 56, "kernel": 2, "stride": 2, "padding": 0},
]


def make_inputs(c, s):
    """Positional arguments for api.max_pool2d, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["n"], c["c"], c["h"], c["w"]), c["kernel"], c["stride"], c["padding"])


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return (c["n"] * c["c"] * (c["h"] * c["w"] + _out(c["h"], c) * _out(c["w"], c))) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None


def _out(size, c):
    return (size + 2 * c["padding"] - c["kernel"]) // c["stride"] + 1
