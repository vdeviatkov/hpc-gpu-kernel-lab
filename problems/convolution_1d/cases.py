"""Test and benchmark cases for 09 · 1D Convolution.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

# Small and tail shapes, K = N (one output), and long filters at and past the K = 255
# bench case where FP32 summation noise is largest relative to the fixed tolerances.
TEST_CASES = [
    {"n": n, "k": k}
    for n, k in [
        (1, 1),
        (5, 5),
        (17, 3),
        (1025, 31),
        (300, 255),
        (1100, 1023),
        (2047, 2047),
        (2100, 2047),
    ]
]
# K = 31 sits at about 8 FLOP/B in FP32 and K = 255 at about 64, either side of the RTX
# 5080's ridge (docs/design.md). N = 2^20 is L2-resident (8 MiB of FP32 traffic against a
# 64 MiB L2), so 25,000,000 with K = 31 is the DRAM-bound counterpart.
BENCH_CASES = [
    {"n": 1048576, "k": 31},
    {"n": 25000000, "k": 31},
    {"n": 25000000, "k": 255},
]


def make_inputs(c, s):
    """Positional arguments for api.conv1d, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["n"]), s.randn(c["k"]))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s: x, w and the output once each."""
    e = element_size(dtype)
    return (2 * c["n"] - c["k"] + 1 + c["k"]) * e


def flops(c):
    """One multiply and one add per filter tap per output."""
    return 2 * (c["n"] - c["k"] + 1) * c["k"]
