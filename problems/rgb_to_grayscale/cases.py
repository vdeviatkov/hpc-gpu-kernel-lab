"""Test and benchmark cases for 05 · RGB to Grayscale.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"h": 1, "w": 1}, {"h": 3, "w": 5}, {"h": 64, "w": 67}]
BENCH_CASES = [{"h": 1080, "w": 1920}, {"h": 4096, "w": 4096}]


def make_inputs(c, s):
    """Positional arguments for api.rgb_to_grayscale, drawn from the lab.data.Sampler `s`."""
    return (s.uniform(0, 1, c["h"], c["w"], 3),)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 4 * c["h"] * c["w"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
