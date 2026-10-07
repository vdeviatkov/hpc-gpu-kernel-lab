"""Run one ReLU backend inside a CUDA profiler capture range.

python -m problems.relu.benchmarks.profile cuda_vec fp16 25000000 mixed

Input generation, extension build and warm-up happen before cudaProfilerStart, so
`ncu --profile-from-start off` or `nsys --capture-range=cudaProfilerApi` captures only
the ReLU launch.
"""

import argparse

import torch

from lab.data import Sampler
from problems.relu import api, cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=api.BACKENDS)
    p.add_argument("dtype", choices=api.DTYPES)
    p.add_argument("n", type=int)
    p.add_argument("dist", choices=["mixed", *cases.DISTS])
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    case = {"n": args.n, "dist": args.dist}
    (x,) = cases.make_inputs(case, Sampler(api.DTYPES[args.dtype], "cuda", args.seed))
    for _ in range(3):
        api.relu(x, backend=args.backend)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        api.relu(x, backend=args.backend)
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
