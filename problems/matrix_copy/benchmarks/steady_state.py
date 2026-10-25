"""Time and profile one backend in lab.bench's steady state, with and without output reuse.

    python -m problems.matrix_copy.benchmarks.steady_state cuda_flat fp16 4096
    ncu --profile-from-start off --cache-control none --metrics ... \\
        python -m problems.matrix_copy.benchmarks.steady_state cuda_flat fp16 4096 --profile

lab.bench keeps the previous call's output alive while the next one is allocated, so two
output blocks alternate and the footprint between calls is 3·N²·e. lab.profile drops each
output at once, so its launches reuse one output block (2·N²·e). This script reproduces
both: `alternating` holds the previous result as lab.bench does; `reused` drops it first.
With --profile, the last two calls of the alternating loop run in a profiler capture range.
"""

import argparse

import torch

from lab import dispatch
from problems.matrix_copy import api


def median_us(fn, keep_previous, warmup=25, samples=100):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    result = None
    times = []
    for i in range(warmup + samples):
        if not keep_previous:
            result = None
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end) * 1000)
    del result
    return sorted(times)[len(times) // 2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("backend")
    p.add_argument("dtype", choices=api.DTYPES)
    p.add_argument("n", type=int)
    p.add_argument("--profile", action="store_true", help="capture two steady-state calls")
    args = p.parse_args()
    a = torch.randn(args.n, args.n, device="cuda").to(api.DTYPES[args.dtype])

    def fn():
        return dispatch.call(api, api.FUNCTION, args.backend, a)

    if args.profile:
        result = None
        for _ in range(25):
            result = fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(2):
            result = fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        del result
        return
    alternating = median_us(fn, keep_previous=True)
    reused = median_us(fn, keep_previous=False)
    print(
        f"{args.backend} {args.dtype} {args.n}: alternating {alternating:.1f} µs, "
        f"reused {reused:.1f} µs"
    )


if __name__ == "__main__":
    main()
