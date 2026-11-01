"""Test and benchmark cases for 08 · Matrix Transpose.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

# Source-valid cases plus empty/oversized lab extensions.
TEST_CASES = [
    {"m": m, "n": n}
    for m, n in [
        (1, 1),
        (2, 3),
        (3, 1),
        (1, 33),
        (3, 5),
        (31, 32),
        (32, 33),
        (65, 129),
        (129, 65),
        (0, 0),
        (0, 5),
        (5, 0),
        (8193, 1),
    ]
]
BENCH_CASES = [
    {"m": m, "n": n} for m, n in [(32, 32), (65, 129), (1024, 1024), (7000, 6000), (8192, 8192)]
]


def make_inputs(c, s):
    """Positional arguments for api.transpose, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["m"], c["n"]),)


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return 2 * c["m"] * c["n"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Not compute-bound.
    return None
