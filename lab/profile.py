"""Run one backend of one problem inside a CUDA profiler capture range.

    python -m lab.profile reverse_array cuda_tile fp16 n=100000001
    ncu --profile-from-start off --metrics ... \\
        python -m lab.profile reverse_array cuda_tile fp16 n=100000001
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
        python -m lab.profile relu cuda_select fp32 n=1025 --launches 100

The case starts from an entry of the problem's BENCH_CASES (by default the last, usually
the largest); key=value arguments override or add keys. Input generation, the extension
build, JIT compilation and warm-up all run before cudaProfilerStart, so a profiler using
the capture range sees only the backend's own launches.

Before profiling, the backend's result is compared with the PyTorch reference, so a
wrong kernel is never profiled; --no-check skips that (and the reference's launches).
"""

import argparse
import importlib

import torch

from lab import dispatch
from lab.data import Sampler, clone_tensors


def parse_value(text):
    """Convert a key=value argument to int, float or bool where it looks like one."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def build_case(bench_cases, overrides, index=-1):
    """BENCH_CASES[index] with key=value overrides applied."""
    case = dict(bench_cases[index])
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"expected key=value, got {item!r}")
        case[key] = parse_value(value)
    return case


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("problem", help="directory name under problems/, e.g. reverse_array")
    p.add_argument("backend", help="a name from the problem's api.BACKENDS")
    p.add_argument("dtype", help="a key of the problem's api.DTYPES, e.g. fp32")
    p.add_argument("overrides", nargs="*", metavar="key=value", help="case keys to set")
    p.add_argument("--case", type=int, default=-1, help="index into BENCH_CASES (default: last)")
    p.add_argument("--launches", type=int, default=1, help="calls inside the capture range")
    p.add_argument("--warmup", type=int, default=3, help="calls before the capture range")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no-check", action="store_true", help="skip the reference comparison")
    return p


def main():
    p = parser()
    args = p.parse_args()
    api = importlib.import_module(f"problems.{args.problem}.api")
    cases = importlib.import_module(f"problems.{args.problem}.cases")
    if args.backend not in api.BACKENDS:
        p.error(f"backend must be one of: {', '.join(api.BACKENDS)}")
    if args.dtype not in api.DTYPES:
        p.error(f"dtype must be one of: {', '.join(api.DTYPES)}")
    if args.launches < 1 or args.warmup < 1:
        p.error("--launches and --warmup must be positive")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        p.error("an NVIDIA CUDA GPU is required")
    try:
        case = build_case(cases.BENCH_CASES, args.overrides, args.case)
    except (ValueError, IndexError) as error:
        p.error(str(error))

    inputs = cases.make_inputs(case, Sampler(api.DTYPES[args.dtype], "cuda", args.seed))

    def call(arguments):
        return dispatch.call(api, api.FUNCTION, args.backend, *arguments)

    if not args.no_check and args.backend != "pytorch":
        expected = dispatch.call(api, api.FUNCTION, "pytorch", *clone_tensors(inputs))
        rtol, atol = api.TOLERANCES[api.DTYPES[args.dtype]]
        torch.testing.assert_close(
            call(clone_tensors(inputs)), expected, rtol=rtol, atol=atol, equal_nan=True
        )
    for _ in range(args.warmup):
        call(clone_tensors(inputs))
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    try:
        for _ in range(args.launches):
            call(inputs)
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    print(f"profiled {args.problem} {args.backend} {args.dtype} {case} x{args.launches}")


if __name__ == "__main__":
    main()
