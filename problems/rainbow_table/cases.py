"""Test and benchmark cases for 06 · Rainbow Table.

Shared by the tests and by `python -m lab.bench`. Each case has a length `n`, a round count
`rounds` and a value distribution `dist`:

- source: integers in [0, 2^31 - 1], the source's distribution;
- full: any int32, including negatives (a lab extension).

Test lengths cover every n mod 4 and mod 8 (vector paths handle 4 elements per thread),
warp and block boundaries; round counts cover 0, 1, small odd counts and the maximum, 100.
"""

from lab.data import element_size

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1

TAIL_SIZES = [
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    15,
    16,
    17,
    31,
    33,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
]
TEST_CASES = (
    [{"n": n, "rounds": 1, "dist": "source"} for n in TAIL_SIZES]
    + [{"n": 1000, "rounds": r, "dist": "source"} for r in (0, 2, 3, 10, 50, 100, 101)]
    # The source's own functional sizes, with negative inputs as well.
    + [
        {"n": 1024, "rounds": 50, "dist": "full"},
        {"n": 10000, "rounds": 10, "dist": "full"},
        {"n": 65537, "rounds": 3, "dist": "full"},
    ]
)
# The source's performance case, a launch-bound size, and a round sweep at the contract's
# maximum N that moves from memory-bound (R = 1: 8 bytes per hash round) to integer-bound
# (R = 100: 0.08 bytes per round).
BENCH_CASES = [
    {"n": 10000, "rounds": 10, "dist": "source"},
    {"n": 5_000_000, "rounds": 10, "dist": "source"},
] + [{"n": 10_000_000, "rounds": r, "dist": "source"} for r in (1, 2, 5, 10, 20, 50, 100)]


def make_inputs(c, s):
    """Positional arguments for api.rainbow_table, drawn from the lab.data.Sampler `s`."""
    low = 0 if c["dist"] == "source" else INT32_MIN
    return (s.randint(low, INT32_MAX + 1, c["n"], dtype=s.dtype), c["rounds"])


def logical_bytes(c, dtype):
    """One int32 read and one uint32 written per element, independent of the round count."""
    return c["n"] * (element_size(dtype) + 4)


def flops(c):
    """Integer operations: 4 XORs and 4 multiplies per round per element.

    Reported in the `tflop_s` field as tera integer-ops per second; divide by 8 for hash
    rounds per second. Byte extraction (shifts and masks) is not counted.
    """
    return 8 * c["n"] * c["rounds"]
