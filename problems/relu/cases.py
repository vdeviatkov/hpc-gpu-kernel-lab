"""Test and benchmark cases for 02 · ReLU.

Shared by the tests and by `python -m lab.bench`. Each case has a size `n` and an
input distribution `dist`:

- mixed: uniform in [-100, 100], the source's performance distribution
- positive / negative: one sign only
- alternating: signs flip between neighbouring elements (and so between lanes)
- warp_blocks: signs flip every 32 elements, so each warp sees one sign
- zeros, extreme, tiny: the source's functional value ranges (0, +-1000, +-0.001)
"""

import torch

from lab.data import element_size

# The source's functional sizes, plus empty input and tails that do not fill a 16-byte pack.
SIZES = [0, 1, 2, 3, 5, 1024, 1025, 10000, 65537]
DISTS = ["positive", "negative", "alternating", "warp_blocks", "zeros", "extreme", "tiny"]

TEST_CASES = [{"n": n, "dist": "mixed"} for n in SIZES] + [
    {"n": n, "dist": d} for d in DISTS for n in (1025, 10000)
]
BENCH_CASES = [{"n": n, "dist": "mixed"} for n in (1025, 1_048_576, 25_000_000)] + [
    {"n": 25_000_000, "dist": d} for d in ("positive", "negative", "alternating", "warp_blocks")
]

RANGES = {
    "mixed": (-100, 100),
    "positive": (0, 100),
    "negative": (-100, 0),
    "zeros": (0, 0),
    "extreme": (-1000, 1000),
    "tiny": (-0.001, 0.001),
}


def make_inputs(c, s):
    """Positional arguments for api.relu, drawn from the lab.data.Sampler `s`."""
    n, dist = c["n"], c["dist"]
    if dist in RANGES:
        low, high = RANGES[dist]
        return (s.uniform(low, high, n, dtype=torch.float32).to(s.dtype),)
    magnitude = s.uniform(0, 100, n, dtype=torch.float32)
    index = torch.arange(n, device=magnitude.device)
    period = 1 if dist == "alternating" else 32  # warp_blocks
    sign = 1 - 2 * ((index // period) % 2)
    return ((magnitude * sign).to(s.dtype),)


def logical_bytes(c, dtype):
    """One read and one write per element, for effective GB/s."""
    e = element_size(dtype)
    return 2 * c["n"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    del c  # One compare per element; bandwidth is the measure.
    return None
