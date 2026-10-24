# Benchmarking

[← Interleave Arrays](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · **Benchmarking** · [Profiling](profiling.md)

Interleave Arrays uses the shared runner in [lab/bench.py](../../../lab/bench.py). Lab-wide
rules are in the [benchmark policy](../../../docs/benchmarking.md).

## Single run

```bash
python -m lab.bench interleave_arrays \
  --backends pytorch pytorch_stack cuda_input cuda_output cuda_pair_store cuda_vec \
  --copy-reference --peak-gb-s 960 --output artifacts/interleave_arrays/device_1.json
python -m lab.report artifacts/interleave_arrays/device_1.json
```

- Each backend is checked against the PyTorch reference before it is timed.
- `--copy-reference` times a device copy of **every** input (here `a` and `b`, one copy
  each) and adds `fraction_of_copy` to every row. That moves exactly the bytes the kernels
  move: both inputs read once, `2N` elements written. At small sizes the two copies are two
  launches, so compare small sizes in µs rather than percent of copy.
- Leave out `--backends` to include `triton`, or use `--backends triton` to measure
  the new kernel alone. The current tables and chart include matched runs of
  every backend, including Triton.

## Benchmark sizes

`BENCH_CASES` in [cases.py](../cases.py): N = 1025 and 1M (launch- and cache-bound),
25,000,000 (the source's performance size) and 50,000,000 (the contract maximum). Out of
place, even 25M FP16 moves 200 MB, far more than the 64 MiB L2, so both large sizes are
DRAM-bound for every dtype.

## Reproducing the current benchmark tables and chart

[sweep.sh](../benchmarks/sweep.sh) now includes Triton. To run the correctness gate
and three complete benchmarks without collecting new Nsight reports:

```bash
BENCH_ONLY=1 PYTHON=.venv/bin/python \
  bash problems/interleave_arrays/benchmarks/sweep.sh artifacts/perf_20261023/interleave_arrays 960
python -m problems.interleave_arrays.benchmarks.plot \
  artifacts/perf_20261023/interleave_arrays/device_*.json
```

The current session is 2026-10-23 PDT / 2026-10-24 UTC on cuda.0: CUDA toolkit
13.4.92, PyTorch 2.14.1+cu130, Triton 3.8.0, driver 615.71.09. Each call gets
25 warmups and 100 samples. Compilation is outside timing; reused inputs,
output allocation and submission gaps are included. The tables and chart use
medians across the three run medians, with all three dtypes in the tables and
FP32/FP16 in the chart. Per-run medians, ranges, metadata and source hashes are
tracked in [measured_summary.json](../benchmarks/measured_summary.json); raw
samples remain in the ignored artifact directory. Timings do not use a CPU fallback.

Omit `BENCH_ONLY=1` to collect a fresh full Nsight sweep as well. The current
results page retains the earlier CUDA/PyTorch profiler measurements separately;
adding benchmark data does not claim new Triton profiler evidence. Plotting needs
matplotlib and can be run locally after copying the JSON files from the GPU host.
