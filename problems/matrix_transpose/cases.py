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
        (40, 520),  # full 16x256 vec_ilp tiles plus tails, pack-aligned for FP32 and FP16
        (0, 0),
        (0, 5),
        (5, 0),
        (8193, 1),
    ]
]
# A size sweep across the RTX 5080's 64 MiB (67.1 MB) L2, the same square sizes as
# 07 · Matrix Copy so the two can be compared shape for shape, plus the source's 7000 x 6000.
# Each call reads the input and writes an output, 2·M·N·e bytes of logical traffic; lab.bench
# keeps the previous output alive while the next is allocated, so the footprint between calls
# is 3·M·N·e. FP32 reaches 64 MiB at N ≈ 2365 (3·N²·e) to 2896 (2·N²·e); FP16/BF16 at
# N ≈ 3344 to 4096.
BENCH_SIZES = [512, 1024, 1536, 2048, 2304, 2560, 2896, 3072, 3584, 4096, 5792, 8192]
SOURCE_CASE = {"m": 7000, "n": 6000}
BENCH_CASES = [{"m": n, "n": n} for n in BENCH_SIZES[:-1]] + [SOURCE_CASE]
BENCH_CASES += [{"m": 8192, "n": 8192}]


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
