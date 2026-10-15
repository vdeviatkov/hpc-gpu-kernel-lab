"""Test and benchmark cases for 05 · RGB to Grayscale.

Shared by the tests and by `python -m lab.bench`. Each case has a height `h`, a width `w`
and a value distribution `dist`:

- int: integers 0-255, the source's performance distribution (exact in every dtype);
- uniform: real values in [0, 255], the source's distribution for its realistic sizes.

Test sizes cover the source's functional images, empty images, and pixel counts with every
remainder mod 8 (a vector path handles 4 FP32 or 8 FP16 pixels per thread and leaves a
scalar tail).
"""

import torch

from lab.data import element_size

SOURCE_SIZES = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (100, 100), (48, 64)]
SOURCE_SIZES += [(3, 2), (2, 3)]  # (h, w): 64 x 48 is width 64, height 48
TAIL_SIZES = [(0, 0), (0, 5), (1, 2), (1, 3), (1, 5), (1, 6), (1, 7), (1, 9)]
TAIL_SIZES += [(3, 5), (5, 7), (17, 33), (127, 129)]

TEST_CASES = [{"h": h, "w": w, "dist": "int"} for h, w in SOURCE_SIZES + TAIL_SIZES] + [
    {"h": h, "w": w, "dist": "uniform"} for h, w in ((48, 64), (100, 100), (17, 33))
]
# 2048 x 2048 is the source's performance size and contract maximum, but its 67 MB barely
# exceeds the 64 MiB L2, so much of the input stays cached between calls; 4096 x 4096
# (268 MB, a lab extension) is DRAM-bound.
BENCH_CASES = [
    {"h": h, "w": w, "dist": "int"} for h, w in ((48, 64), (1024, 1024), (2048, 2048), (4096, 4096))
]


def make_inputs(c, s):
    """Positional arguments for api.rgb_to_grayscale, drawn from the lab.data.Sampler `s`."""
    shape = (c["h"], c["w"], 3)
    if c["dist"] == "int":
        values = s.randint(0, 256, *shape, dtype=torch.float32)
    else:
        values = s.uniform(0, 255, *shape, dtype=torch.float32)
    return (values.to(s.dtype),)


def logical_bytes(c, dtype):
    """Three channels read and one value written per pixel."""
    e = element_size(dtype)
    return 4 * c["h"] * c["w"] * e


def flops(c):
    """Three multiplies and two adds per pixel; bandwidth, not arithmetic, is the limit."""
    return 5 * c["h"] * c["w"]
