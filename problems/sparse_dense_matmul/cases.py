"""Test and benchmark cases for 28 · Sparse Matrix-Dense Matrix Multiplication.

Shared by the tests and by `python -m lab.bench`.
"""

from lab.data import element_size

TEST_CASES = [
    {"m": 1, "k": 1, "n": 1, "nnz_per_row": 1},
    {"m": 33, "k": 65, "n": 17, "nnz_per_row": 4},
]
BENCH_CASES = [
    {"m": 16384, "k": 16384, "n": 256, "nnz_per_row": 32},
    {"m": 65536, "k": 65536, "n": 64, "nnz_per_row": 16},
]


def make_inputs(c, s):
    """Positional arguments for api.spmm, drawn from the lab.data.Sampler `s`."""
    return (*s.csr(c["m"], c["k"], c["nnz_per_row"]), s.randn(c["k"], c["n"]))


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    e = element_size(dtype)
    return c["m"] * c["nnz_per_row"] * (4 + e) + c["k"] * c["n"] * e + c["m"] * c["n"] * e


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return 2 * c["m"] * c["nnz_per_row"] * c["n"]
