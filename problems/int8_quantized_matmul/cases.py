"""Test and benchmark cases for 27 · INT8 Quantized MatMul.

Shared by the tests and by `python -m lab.bench`.
"""

import torch

TEST_CASES = [{"m": 1, "k": 1, "n": 1}, {"m": 3, "k": 5, "n": 7}, {"m": 65, "k": 33, "n": 129}]
BENCH_CASES = [{"m": 1024, "k": 1024, "n": 1024}, {"m": 8192, "k": 8192, "n": 8192}]


def make_inputs(c, s):
    """Positional arguments for api.int8_matmul, drawn from the lab.data.Sampler `s`."""
    return (
        s.randint(-128, 128, c["m"], c["k"], dtype=torch.int8),
        s.randint(-128, 128, c["k"], c["n"], dtype=torch.int8),
        0.02,
        0.03,
        3,
        -5,
    )


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    del dtype  # Byte count is dtype-independent.
    return c["m"] * c["k"] + c["k"] * c["n"] + 4 * c["m"] * c["n"]


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 2 * c["m"] * c["n"] * c["k"]
