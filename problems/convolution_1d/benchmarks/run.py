"""Run lab.bench over one of this study's sweeps instead of cases.BENCH_CASES.

python -m problems.convolution_1d.benchmarks.run ksweep --backends cuda_naive triton \\
    --dtypes fp32 fp16 --output artifacts/convolution_1d/ksweep_1.json

Sweeps (the first argument; the rest are lab.bench options):

- ksweep  N = 2^24, K in KSWEEP: the roofline figure. FP32 intensity runs from 0.25 to
          512 FLOP/B, across the ridge at about 64.
- nsweep  K = 31, N in NSWEEP: effective bandwidth against working-set size, across the
          64 MiB L2 (the transpose L2 cliff, for a stencil).
- rsweep  N = 2^24, K in RSWEEP_K, with cuda_smem_reg_r{2,4,8,16}: cuda_smem_reg at each
          compiled R (outputs per thread), registered as extra backends for this run only.
- copy    07 · Matrix Copy on (N / 1024) x 1024 matrices for every N above: 2·N·e bytes, the
          same logical traffic as conv1d's (2N + 1)·e, and the bandwidth ceiling. Not a
          convolution and never compared with the reference.

Every timed backend is first checked against the PyTorch reference by lab.bench.
"""

import sys
from pathlib import Path

from lab import bench
from problems.convolution_1d import api, cases

N_KSWEEP = 1 << 24
KSWEEP = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]
MIB = 1 << 20
# FP32 conv1d keeps x plus two outputs alive between lab.bench samples (the previous output
# is still referenced while the next is allocated): about 3·N·e bytes. That reaches the
# 64 MiB L2 at N ≈ 5.6 M (FP32) and 11.2 M (FP16); 2·N·e, the traffic of one call, at
# 8.4 M and 16.8 M.
NSWEEP = [256 * 1024, 512 * 1024] + [m * MIB for m in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14)]
NSWEEP += [m * MIB for m in (16, 20, 24, 32, 64)]
RSWEEP_K = [31, 255, 2047]
REG_OUTPUTS = (2, 4, 8, 16)

SWEEPS = {
    "ksweep": [{"n": N_KSWEEP, "k": k} for k in KSWEEP],
    "nsweep": [{"n": n, "k": 31} for n in NSWEEP],
    "rsweep": [{"n": N_KSWEEP, "k": k} for k in RSWEEP_K],
}


def copy_cases():
    sizes = sorted(set(NSWEEP) | {N_KSWEEP})
    assert all(n % 1024 == 0 for n in sizes)
    return [{"m": n // 1024, "n": 1024} for n in sizes]


def register_reg_variants():
    """cuda_smem_reg_r{R} backends, for the R sweep only; the api itself is unchanged."""
    names = []
    for r in REG_OUTPUTS:
        name = f"cuda_smem_reg_r{r}"
        api.VARIANTS[name] = ("cuda", {"variant": 3, "reg_outputs": r})
        names.append(name)
    api.BACKENDS = (*api.BACKENDS, *names)
    return names


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in (*SWEEPS, "copy"):
        raise SystemExit(f"usage: run.py {{{','.join((*SWEEPS, 'copy'))}}} [lab.bench options]")
    sweep, rest = sys.argv[1], sys.argv[2:]
    if sweep == "copy":
        from problems.matrix_copy import cases as matrix_copy_cases

        args = bench.parser().parse_args(["matrix_copy", *rest])
        matrix_copy_cases.BENCH_CASES = copy_cases()
    else:
        args = bench.parser().parse_args(["convolution_1d", *rest])
        if sweep == "rsweep":
            names = register_reg_variants()
            args.backends = args.backends or names
        cases.BENCH_CASES = SWEEPS[sweep]
    args.output = args.output or Path(f"artifacts/convolution_1d/{sweep}.json")
    bench.run_problem(args)


if __name__ == "__main__":
    main()
