# Benchmarking

[← Matrix Copy](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · **Benchmarking** · [Profiling](profiling.md)

Matrix Copy uses the shared runner in [lab/bench.py](../../../lab/bench.py). Lab-wide rules
are in the [benchmark policy](../../../docs/benchmarking.md).

## Reproduce

On cuda.0, from `/home/vdeviatkov/Documents/hpc-gpu-kernel-lab`:

```bash
PYTHON=.venv/bin/python bash problems/matrix_copy/benchmarks/sweep.sh artifacts/matrix_copy 960
# where matplotlib is installed, after copying artifacts/matrix_copy:
python -m problems.matrix_copy.benchmarks.plot artifacts/matrix_copy
```

[sweep.sh](../benchmarks/sweep.sh) runs, in order:

1. The full correctness suite (`tests.log`). A failure stops the sweep.
2. **Three complete runs** of `lab.bench` over every backend plus `--copy-reference`
   (`device_{1,2,3}.json`).
3. 40 Nsight Compute reports (see [profiling](profiling.md)) and their raw CSV exports.
4. Static SASS of the CUDA extension and both Triton kernels, with per-kernel load/store
   counts (`sass/counts.json`).
5. [summarize.py](../benchmarks/summarize.py), which writes the tracked
   [measured_summary.json](../benchmarks/measured_summary.json) and prints the tables in
   [results](results.md). `SKIP_BENCH=1` reuses existing benchmark JSON and only
   re-profiles.

## What one sample measures

- Each backend is checked against the reference before it is timed. Backend order is
  shuffled per case with a fixed seed.
- 25 warmups, then 100 samples. One sample is one Python call between two CUDA events:
  argument checks, output allocation, launch and kernel. The fixed part is about 10 µs,
  so at small sizes the samples measure the call, not the kernel. Kernel-only durations
  come from [profiling](profiling.md).
- `effective_gb_s` is `2·N²·e` bytes over the median latency. Above 960 GB/s means L2
  served part of the traffic.
- `copy_reference` is `clone` of the input timed by the harness: the same operation as the
  `pytorch` backend, measured separately as a control. `fraction_of_copy` is the copy's
  latency over the backend's, so 1.0 means as fast as memcpy.
- Inputs are reused without flushing, and two output blocks alternate between calls
  ([design](design.md#work-and-traffic)). Clocks are unlocked.

## Sizes

`BENCH_CASES` in [cases.py](../cases.py): N = 512, 1024, 1536, 2048, 2304, 2560, 2896,
3072, 3584, 4096, 5792, 8192, square, standard-normal values (a copy's speed does not
depend on them). The dense steps between 2048 and 4096 place the L2 boundary for FP32 and
FP16; 4096 is the source's performance size; 8192 is a lab extension that streams
537 MB (FP32) per call.

Raw samples and full environments stay in the ignored `artifacts/matrix_copy/`;
per-run medians, metadata, profiler metrics and SASS counts are tracked in
[measured_summary.json](../benchmarks/measured_summary.json).
