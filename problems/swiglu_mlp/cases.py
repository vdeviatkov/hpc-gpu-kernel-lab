"""Test and benchmark cases for 39 · SwiGLU MLP Block.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [{"m": 1, "d": 1, "f": 1}, {"m": 3, "d": 8, "f": 16}, {"m": 33, "d": 64, "f": 176}]
BENCH_CASES = [{"m": 4096, "d": 4096, "f": 11008}, {"m": 64, "d": 4096, "f": 11008}]


def make_inputs(c, s):
    """Positional arguments for api.swiglu_mlp, drawn from the lab.data.Sampler `s`."""
    d, f = c["d"], c["f"]
    return (
        s.randn(c["m"], d),
        s.randn(d, f, scale=d**-0.5),
        s.randn(d, f, scale=d**-0.5),
        s.randn(f, d, scale=f**-0.5),
    )


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return (2 * c["m"] * c["d"] + 3 * c["d"] * c["f"]) * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 6 * c["m"] * c["d"] * c["f"]
