"""Test and benchmark cases for 22 · Sparse Matrix-Vector Multiplication.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [
    {"m": 1, "k": 1, "nnz_per_row": 1},
    {"m": 17, "k": 33, "nnz_per_row": 4},
    {"m": 257, "k": 513, "nnz_per_row": 16},
]
BENCH_CASES = [
    {"m": 100000, "k": 100000, "nnz_per_row": 16},
    {"m": 1000000, "k": 1000000, "nnz_per_row": 8},
]


def make_inputs(c, s):
    """Positional arguments for api.spmv, drawn from the lab.data.Sampler `s`."""
    return (*s.csr(c["m"], c["k"], c["nnz_per_row"]), s.randn(c["k"]))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return c["m"] * c["nnz_per_row"] * (4 + 2 * e) + (c["m"] + 1) * 4 + c["m"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 2 * c["m"] * c["nnz_per_row"]
