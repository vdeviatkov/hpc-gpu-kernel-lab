"""Test and benchmark cases for 07 · Matrix Copy.

Shared by the tests and by `python -m lab.bench`. Each case is an M x N matrix; `m == n`
is the source's square contract, anything else is a lab extension.

Square test sizes cover every N mod 4 and mod 8 (a 16-byte pack holds four FP32 or eight
FP16/BF16 elements, so a row length that is not a multiple leaves a tail), warp and block
boundaries, and N = 1025, just past a power of two. Rectangular cases put the tail in the
row (n) or only in the row count (m), and the empty cases have no elements at all.
600000 x 1 and 1 x 600000 need more than 65535 row (column) tiles of 8: the 2-D CUDA
kernels must loop over grid axis y. N = 1025 has more packs than one wave of resident
threads on current GPUs, so the grid-stride loop iterates.
"""

from lab.data import element_size

SQUARE_SIZES = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257]
SQUARE_SIZES += [1023, 1024, 1025]
RECTANGULAR = [(1, 1000), (1000, 1), (3, 5), (5, 3), (65, 129), (129, 65), (7, 4099)]
RECTANGULAR += [(600_000, 1), (1, 600_000)]
EMPTY = [(0, 0), (0, 7), (7, 0)]

TEST_CASES = (
    [{"m": n, "n": n} for n in SQUARE_SIZES]
    + [{"m": m, "n": n} for m, n in RECTANGULAR]
    + [{"m": m, "n": n} for m, n in EMPTY]
)
# A size sweep across the RTX 5080's 64 MiB (67.1 MB) L2. Each call reads the input and
# writes an output, 2·N²·e bytes of logical traffic. lab.bench reuses the input, but the
# previous call's output is still referenced while the next is allocated, so two output
# blocks alternate: the footprint between calls is 3·N²·e. FP32 reaches 64 MiB at N ≈ 2365
# (3·N²·e) to 2896 (2·N²·e); FP16/BF16 at N ≈ 3344 to 4096. 4096 is the source's
# performance size; 8192 is a lab extension.
BENCH_SIZES = [512, 1024, 1536, 2048, 2304, 2560, 2896, 3072, 3584, 4096, 5792, 8192]
BENCH_CASES = [{"m": n, "n": n} for n in BENCH_SIZES]


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
