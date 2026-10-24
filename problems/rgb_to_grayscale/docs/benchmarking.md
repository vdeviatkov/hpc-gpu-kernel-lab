# Benchmarking

[← RGB to Grayscale](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · **Benchmarking** · [Profiling](profiling.md)

RGB to Grayscale uses the shared runner in [lab/bench.py](../../../lab/bench.py). Lab-wide
rules are in the [benchmark policy](../../../docs/benchmarking.md).

## Single run

```bash
python -m lab.bench rgb_to_grayscale \
  --backends pytorch pytorch_matmul pytorch_compile \
             cuda_pixel cuda_vec cuda_smem cuda_planar \
  --copy-reference --peak-gb-s 960 --output artifacts/rgb_to_grayscale/device_1.json
python -m lab.report artifacts/rgb_to_grayscale/device_1.json
```

- Each backend is checked against the PyTorch reference before it is timed.
- One sample is one Python call, timed with CUDA events: argument checks, output
  allocation, launch and kernel. That fixed part is about 11 µs for the CUDA extension
  and 19.5 µs for `pytorch_compile` (the time at 48 × 64), so at small and cache-resident
  sizes the samples measure the call, not the kernel. Kernel-only times come from
  [profiling](profiling.md).
- `--copy-reference` times a device copy of the image (`6·P·s` bytes moved, a 1:1
  read/write mix). The kernel moves `4·P·s` bytes, 3:1, so compare GB/s with the copy
  rather than the time or `fraction_of_copy`.
- `effective_gb_s` is the logical traffic (`4·P·s`: the image read once, the output
  written once) per second. Above the DRAM peak means L2 served part of it.
- `pytorch_compile` compiles once per shape and dtype during the warmup; compilation is
  never timed.
- Leave out `--backends` to include `triton`, or use `--backends triton` to measure
  the new kernel alone. The current tables and chart include matched runs of
  every backend, including Triton.

## Benchmark sizes

`BENCH_CASES` in [cases.py](../cases.py), integer values 0–255:

| Size | FP32 traffic | Regime |
|---|---:|---|
| 48 × 64 | 49 KB | the source's example size: launch- and call-bound |
| 1024 × 1024 | 17 MB | L2-resident in every dtype |
| 2048 × 2048 | 67 MB (FP16: 34 MB) | the source's performance size: FP16 stays in L2, FP32 just exceeds it |
| 4096 × 4096 | 268 MB (FP16: 134 MB) | lab extension: DRAM-bound in every dtype |

## Reproducing the current benchmark tables and chart

[sweep.sh](../benchmarks/sweep.sh) now includes Triton. To run the correctness gate
and three complete benchmarks without collecting new Nsight reports:

```bash
BENCH_ONLY=1 PYTHON=.venv/bin/python \
  bash problems/rgb_to_grayscale/benchmarks/sweep.sh artifacts/perf_20261023/rgb_to_grayscale 960
python -m problems.rgb_to_grayscale.benchmarks.plot \
  artifacts/perf_20261023/rgb_to_grayscale/device_*.json
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
