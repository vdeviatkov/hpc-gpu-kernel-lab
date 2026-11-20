"""Test and benchmark cases for 10 · Gaussian Blur.

Shared by the tests and by `python -m lab.bench`.

A case is (h, w) image rows and columns, (kh, kw) kernel rows and columns, and a kernel
kind:

- "gaussian": a sampled, normalized 2-D Gaussian with sigma = k/6 per axis, built in FP64
  as the outer product of two 1-D Gaussians and rounded once to the input dtype, so it is
  separable up to that rounding.
- "random": the source's performance-case kernel, uniform in [0.0001, 0.02) and not
  normalized. Its rank-1 residual is 20-46% of its norm, so it is not separable and
  exercises the direct path (docs/design.md, "Separability").
"""

import torch

from lab.data import element_size


def _case(h, w, kh, kw, kernel="gaussian"):
    return {"h": h, "w": w, "kh": kh, "kw": kw, "kernel": kernel}


TEST_CASES = [
    _case(1, 1, 1, 1),  # source single_pixel test
    _case(1, 1, 3, 3),  # kernel larger than the image
    _case(1, 1, 21, 21, "random"),
    _case(5, 5, 3, 3),  # source example shape
    _case(9, 13, 3, 5),  # rectangular kernel, odd W
    _case(13, 9, 5, 3, "random"),
    _case(17, 31, 1, 7),  # a 1-D kernel along each axis
    _case(31, 17, 7, 1),
    _case(2, 37, 21, 21),  # kernel taller than the image, odd W
    _case(64, 67, 7, 7),  # the source performance kernel size
    _case(64, 67, 7, 7, "random"),
    _case(33, 65, 21, 21),  # the source's largest kernel
    _case(33, 65, 21, 21, "random"),
    _case(37, 41, 31, 31),  # past the source limit: 961 taps of FP32 summation
    _case(37, 41, 31, 31, "random"),
]
# The source performance case first: 512 x 512 FP32 is 1 MiB in and 1 MiB out, so it is
# L2-resident and launch-latency-sized on the RTX 5080. 4096 x 4096 is the source's
# largest image (64 MiB of FP32 traffic, about the L2 size), at the source's largest
# kernel, separable and not.
BENCH_CASES = [
    _case(512, 512, 7, 7, "random"),
    _case(4096, 4096, 7, 7),
    _case(4096, 4096, 21, 21),
    _case(4096, 4096, 21, 21, "random"),
]


def gaussian_1d(k, sigma=None):
    """Normalized samples of exp(-x^2 / (2 sigma^2)) at x = -k//2 .. k//2, in FP64."""
    x = torch.arange(k, dtype=torch.float64) - k // 2
    sigma = k / 6 if sigma is None else sigma
    g = torch.exp(-x * x / (2 * sigma * sigma))
    return g / g.sum()


def gaussian_2d(kh, kw):
    """Separable 2-D Gaussian in FP64: outer(gaussian_1d(kh), gaussian_1d(kw))."""
    return torch.outer(gaussian_1d(kh), gaussian_1d(kw))


def make_kernel(c, s):
    if c["kernel"] == "gaussian":
        return gaussian_2d(c["kh"], c["kw"]).to(s.dtype).to(s.device)
    if c["kernel"] == "random":
        return s.uniform(0.0001, 0.02, c["kh"], c["kw"])
    raise ValueError(f"unknown kernel kind: {c['kernel']}")


def make_inputs(c, s):
    """Positional arguments for api.gaussian_blur, drawn from the lab.data.Sampler `s`."""
    return (s.randn(c["h"], c["w"]), make_kernel(c, s))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s: image, kernel, output once."""
    e = element_size(dtype)
    return (2 * c["h"] * c["w"] + c["kh"] * c["kw"]) * e


def flops(c):
    """Useful work of the direct 2-D filter, 2 * kh * kw per pixel, whichever algorithm
    runs: a separable backend does 2 * (kh + kw) per pixel and reports the direct count,
    so its TFLOP/s is an effective rate (docs/design.md)."""
    return 2 * c["h"] * c["w"] * c["kh"] * c["kw"]
