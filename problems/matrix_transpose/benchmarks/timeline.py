"""Drive every transpose kernel and the copy controls for an Nsight Systems timeline.

nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \\
    -o artifacts/matrix_transpose/nsys/timeline \\
    python -m problems.matrix_transpose.benchmarks.timeline

Each backend gets one NVTX range per dtype ("transpose/cuda_smem fp32", "copy/cuda_vec fp32",
...) holding --launches calls on the source's 7000 x 6000 case. Builds, JIT compilation and
warm-up happen before cudaProfilerStart, so the capture holds only the timed calls.
"""

import argparse

import torch

from lab import dispatch
from lab.data import Sampler
from problems.matrix_copy import api as copy_api
from problems.matrix_transpose import api, cases

BACKENDS = ("pytorch", *api.VARIANTS, "triton")
COPY_BACKENDS = ("pytorch", "cuda_vec")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--launches", type=int, default=10)
    p.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"])
    args = p.parse_args()
    runs = [(api, "transpose", b) for b in BACKENDS]
    runs += [(copy_api, "copy", b) for b in COPY_BACKENDS]
    inputs = {
        d: cases.make_inputs(cases.SOURCE_CASE, Sampler(api.DTYPES[d], "cuda"))[0]
        for d in args.dtypes
    }
    for module, _, backend in runs:
        for a in inputs.values():
            for _ in range(3):
                dispatch.call(module, module.FUNCTION, backend, a)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    try:
        for dtype, a in inputs.items():
            for module, label, backend in runs:
                torch.cuda.nvtx.range_push(f"{label}/{backend} {dtype}")
                for _ in range(args.launches):
                    dispatch.call(module, module.FUNCTION, backend, a)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
    finally:
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
