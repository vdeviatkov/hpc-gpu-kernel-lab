"""Test and benchmark cases for 06 · Rainbow Table.

Shared by the tests and by `python -m lab.bench`.
"""

TEST_CASES = [{"n": 1, "rounds": 1}, {"n": 1025, "rounds": 3}, {"n": 65537, "rounds": 10}]
BENCH_CASES = [{"n": 1048576, "rounds": 10}, {"n": 25000000, "rounds": 100}]


def make_inputs(c, s):
    """Positional arguments for api.rainbow_table, drawn from the lab.data.Sampler `s`."""
    return (s.randint(0, 2**31 - 1, c["n"]), c["rounds"])


def logical_bytes(c, dtype):
    """Minimum bytes read and written, for effective GB/s."""
    del dtype  # Byte count is dtype-independent.
    return 2 * c["n"] * 4


def flops(c):
    """Arithmetic operations, for TFLOP/s; None when bandwidth is the measure."""
    return c["n"] * c["rounds"]
