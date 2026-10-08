"""Test and benchmark cases for 03 · Reverse Array.

Shared by the tests and by `python -m lab.bench`. Each case has a size `n` and a value
range `dist`; reversal only moves values, so the sizes matter more than the values:

- odd and even N (the middle element of odd N must stay in place);
- N around warp (32) and block-boundary sizes: with 256 threads per block, one thread
  per pair, N = 512 is exactly one block of pairs and 513/514 spill into a second;
- every N mod 8, so a vector path sees each alignment of the mirrored half.
"""

import torch

from lab.data import element_size

SIZES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 30, 31, 32, 33, 63, 64, 65, 511, 512, 513, 514]
SIZES += [1000, 1023, 1024, 1025, 10000, 65535, 65536, 65537]

# The source's functional value ranges (in addition to its sizes, which SIZES covers).
RANGES = {
    "uniform": (-1000, 1000),  # the source's performance distribution
    "zeros": (0, 0),
    "unit": (0, 1),
    "tiny": (-1e-6, 1e-6),
    "large": (-1e7, 1e7),
}

TEST_CASES = [{"n": n, "dist": "uniform"} for n in SIZES] + [
    {"n": n, "dist": d} for d in ("zeros", "unit", "tiny", "large") for n in (4, 1000, 10000)
]
# 25,000,000 is a multiple of 8; 25,000,001 and 25,000,002 misalign the mirrored half
# for 16-byte vector accesses.
BENCH_CASES = [
    {"n": n, "dist": "uniform"} for n in (1025, 1_048_576, 25_000_000, 25_000_001, 25_000_002)
]


def make_inputs(c, s):
    """Positional arguments for api.reverse_, drawn from the lab.data.Sampler `s`."""
    low, high = RANGES[c["dist"]]
    # Keep values finite in FP16 (max 65504): +-1e7 would otherwise become +-inf.
    limit = torch.finfo(s.dtype).max
    low, high = max(low, -limit), min(high, limit)
    return (s.uniform(low, high, c["n"], dtype=torch.float32).to(s.dtype),)


def logical_bytes(c, dtype):
    """Every element except the middle one of odd N is read once and written once."""
    e = element_size(dtype)
    moved = 2 * (c["n"] // 2)
    return 2 * moved * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Pure data movement.
    return None
