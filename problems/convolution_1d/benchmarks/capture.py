"""One checked launch inside a CUDA profiler capture range, for Nsight Compute.

ncu --profile-from-start off --metrics ... \\
    python -m problems.convolution_1d.benchmarks.capture cuda_smem_reg fp32 16777216 255 \\
    --reg-outputs 4

Like lab.profile, but takes N and K directly, passes cuda_smem_reg's reg_outputs (R), and
can capture 07 · Matrix Copy's cuda_vec on an (N / 1024) x 1024 matrix (`copy_cuda_vec`),
the bandwidth control. Conv1d backends are checked against the PyTorch reference first;
inputs, compilation, the reference and warm-up all happen before cudaProfilerStart.
"""

import argparse

import torch

from lab.data import Sampler
from problems.convolution_1d import api


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("backend", help="a conv1d backend, or copy_cuda_vec")
    p.add_argument("dtype", choices=list(api.DTYPES))
    p.add_argument("n", type=int)
    p.add_argument("k", type=int)
    p.add_argument("--reg-outputs", type=int, default=None)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()
    dtype = api.DTYPES[args.dtype]
    s = Sampler(dtype, "cuda", seed=2026)

    if args.backend == "copy_cuda_vec":
        from problems.matrix_copy import api as copy_api

        a = s.randn(args.n // 1024, 1024)

        def call():
            return copy_api.matrix_copy(a, backend="cuda_vec")

    else:
        x, w = s.randn(args.n), s.randn(args.k)
        options = {} if args.reg_outputs is None else {"reg_outputs": args.reg_outputs}

        def call():
            return api.conv1d(x, w, backend=args.backend, **options)

        rtol, atol = api.TOLERANCES[dtype]
        torch.testing.assert_close(call(), api.conv1d(x, w), rtol=rtol, atol=atol)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    call()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"captured {args.backend} {args.dtype} n={args.n} k={args.k} {options_label(args)}")


def options_label(args):
    return "" if args.reg_outputs is None else f"reg_outputs={args.reg_outputs}"


if __name__ == "__main__":
    main()
