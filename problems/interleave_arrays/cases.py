"""Test and benchmark cases for 04 · Interleave Arrays.

Shared by the tests and by `python -m lab.bench`. Each case has a length `n` and a value
distribution `dist`:

- normal: standard normal, the source's performance distribution;
- ordered: a[i] = 2i and b[i] = 2i + 1, so a correct result is 0, 1, 2, ... and any
  swapped, shifted or duplicated element is visible (exact for small n in every dtype).

Sizes cover the source's functional lengths, every n mod 8 (a 16-byte pack holds 4 FP32
or 8 FP16/BF16 elements, so vector paths leave a scalar tail), and warp/block boundaries.
"""

import torch

from lab.data import element_size

SIZES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256, 257]
SIZES += [1024, 1025, 10000, 65537, 100000]

TEST_CASES = [{"n": n, "dist": "normal"} for n in SIZES] + [
    # Ordered values stay exact up to n = 128 in BF16 (outputs up to 255).
    {"n": n, "dist": "ordered"}
    for n in (1, 5, 17, 33, 128)
]
# 25M is the source's performance size; 50M is the contract maximum. Out of place, even
# 25M FP16 (200 MB moved) is far larger than the 64 MiB L2.
BENCH_CASES = [{"n": n, "dist": "normal"} for n in (1025, 1_048_576, 25_000_000, 50_000_000)]


def make_inputs(c, s):
    """Positional arguments for api.interleave, drawn from the lab.data.Sampler `s`."""
    n = c["n"]
    if c["dist"] == "ordered":
        even = torch.arange(0, 2 * n, 2, dtype=torch.float32)
        return (even.to(s.dtype).to(s.device), (even + 1).to(s.dtype).to(s.device))
    return (s.randn(n), s.randn(n))


def logical_bytes(c, dtype):
    """Both inputs read once, the 2N-element output written once."""
    e = element_size(dtype)
    return 4 * c["n"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # Pure data movement.
    return None
