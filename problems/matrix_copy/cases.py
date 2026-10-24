"""Test and benchmark cases for 07 · Matrix Copy.

Shared by the tests and by `python -m lab.bench`. Each case is an M x N matrix; `m == n`
is the source's square contract, anything else is a lab extension.

Square test sizes cover every N mod 4 and mod 8 (a 16-byte pack holds four FP32 or eight
FP16/BF16 elements, so a row length that is not a multiple leaves a tail), warp and block
boundaries, and N = 1025, just past a power of two. Rectangular cases put the tail in the
row (n) or only in the row count (m), and the empty cases have no elements at all.
"""

from lab.data import element_size

SQUARE_SIZES = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257]
SQUARE_SIZES += [1023, 1024, 1025]
RECTANGULAR = [(1, 1000), (1000, 1), (3, 5), (5, 3), (65, 129), (129, 65), (7, 4099)]
EMPTY = [(0, 0), (0, 7), (7, 0)]

TEST_CASES = (
    [{"m": n, "n": n} for n in SQUARE_SIZES]
    + [{"m": m, "n": n} for m, n in RECTANGULAR]
    + [{"m": m, "n": n} for m, n in EMPTY]
)
# Total traffic (read + write, FP32) against the RTX 5080's 64 MB L2: 1024² moves 8 MB
# and 2048² 32 MB, both small enough to stay L2-resident across repeated launches
# (lab.bench reuses inputs without flushing); 4096², the source's performance case, moves
# 128 MB and 8192² (a lab extension) 512 MB, so those stream from DRAM.
BENCH_CASES = [{"m": n, "n": n} for n in (1024, 2048, 4096, 8192)]


def make_inputs(c, s):
    """Positional arguments for api.matrix_copy, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["m"], c["n"]),)


def logical_bytes(c, dtype):
    """Every element read once and written once."""
    return 2 * c["m"] * c["n"] * element_size(dtype)


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # A copy does no arithmetic.
    return None
