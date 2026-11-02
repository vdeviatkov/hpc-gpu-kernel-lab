# Benchmarking

[← Matrix Transpose](../README.md) · [Design](design.md) · [Results](results.md) · [Testing](testing.md) ·
**Benchmarking** · [Profiling](profiling.md)

Matrix Transpose uses the shared runner in [lab/bench.py](../../../lab/bench.py). Lab-wide
rules are in the [benchmark policy](../../../docs/benchmarking.md).

## Reproduce

On cuda.0, from a checkout of this repository:

```bash
PYTHON=.venv/bin/python bash problems/matrix_transpose/benchmarks/sweep.sh artifacts/matrix_transpose 960
# where matplotlib is installed, after copying artifacts/matrix_transpose:
python -m problems.matrix_transpose.benchmarks.plot artifacts/matrix_transpose
```

[sweep.sh](../benchmarks/sweep.sh) runs, in order:

1. The full correctness suite (`tests.log`), then Compute Sanitizer memcheck, racecheck,
   synccheck and initcheck over the CUDA and Triton tests (`sanitizer_*.log`; see
   [testing](testing.md)). A failure stops the sweep.
2. **Three complete runs** of `lab.bench` over `pytorch`, the six CUDA kernels and `triton` with
   `--copy-reference` (`device_{1,2,3}.json`), each followed by
   [copy_control.py](../benchmarks/copy_control.py): 07 · Matrix Copy's `cuda_vec` and
   memcpy on the same cases (`copy_{1,2,3}.json`), the plain-copy ceilings.
3. 32 Nsight Compute reports with selected metrics and 14 full-set reports
   (see [profiling](profiling.md)), with raw CSV exports.
4. An Nsight Systems timeline of every kernel and both copy controls
   ([timeline.py](../benchmarks/timeline.py)), with its kernel and NVTX summaries.
5. Static SASS and resource usage of the CUDA extension (`sass/`). Triton's SASS is in its
   full Nsight Compute reports (Source page).
6. [summarize.py](../benchmarks/summarize.py), which writes the tracked
   [measured_summary.json](../benchmarks/measured_summary.json) and prints the tables in
   [results](results.md). `SKIP_BENCH=1` reuses existing benchmark JSON and only re-profiles.

## What one sample measures

- Each backend is checked against the PyTorch reference before it is timed. Backend order
  is shuffled per case with a fixed seed.
- 25 warmups, then 100 samples. One sample is one Python call between two CUDA events:
  argument checks, output allocation, launch and kernel. The fixed part is about 10 µs
  (CUDA extension) or 15 µs (Triton), so at small sizes the samples measure the call, not
  the kernel. Kernel-only durations come
  from [profiling](profiling.md).
- `effective_gb_s` is `2·M·N·e` bytes over the median latency. Above 960 GB/s means L2
  served part of the traffic.
- **Copy ceilings, not competitors.** `copy_reference` (shown as `memcpy`) is `a.clone()`
  timed by the harness ([lab/bench.py](../../../lab/bench.py), `time_copy_reference`), a
  driver device-to-device copy. The copy control times 07's `cuda_vec`, a 16-byte pack copy
  kernel ([kernels.cu](../../matrix_copy/cuda/kernels.cu), `pack_kernel`). Both are plain
  contiguous copies, **not transposes**, and neither is checked against the transposed
  reference. A copy and a transpose of one shape move the same logical bytes, so a
  transpose can at best match them; every table groups them under *Copy ceilings*, apart
  from the *Transposes*.
- Inputs are reused without flushing, and two output blocks alternate between calls
  ([cases.py](../cases.py)). Clocks are unlocked.

## Sizes

`BENCH_CASES` in [cases.py](../cases.py): square N = 512 to 5792, the source's 7000 × 6000,
and 8192², standard-normal values. The square sizes are 07's, so the copy control lines up
shape for shape; the dense steps between 2048 and 4096 place the L2 boundary for FP32 and
FP16. BF16 uses the same 2-byte kernels as FP16 and is not timed separately.

Raw samples, full environments and profiler reports stay in the ignored
`artifacts/matrix_transpose/`; per-run medians, metadata, profiler metrics and the timeline
kernel summary are tracked in [measured_summary.json](../benchmarks/measured_summary.json).
