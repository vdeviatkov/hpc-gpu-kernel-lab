"""Time 07 · Matrix Copy on this problem's BENCH_CASES: the bandwidth control for transpose.

python -m problems.matrix_transpose.benchmarks.copy_control --backends pytorch cuda_vec \\
    --copy-reference --output artifacts/matrix_transpose/copy_1.json

Takes lab.bench's options. A copy and a transpose of the same shape and dtype move the same
logical bytes, so their latencies compare directly; only the case list is replaced.
"""

import sys
from pathlib import Path

from lab import bench
from problems.matrix_copy import cases as copy_cases
from problems.matrix_transpose import cases


def main():
    args = bench.parser().parse_args(["matrix_copy", *sys.argv[1:]])
    args.output = args.output or Path("artifacts/matrix_transpose/copy.json")
    copy_cases.BENCH_CASES = cases.BENCH_CASES
    bench.run_problem(args)


if __name__ == "__main__":
    main()
