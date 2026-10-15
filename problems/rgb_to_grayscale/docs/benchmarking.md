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
- Leave out `--backends` to include `triton`, which reports "not implemented".

## Benchmark sizes

`BENCH_CASES` in [cases.py](../cases.py), integer values 0–255:

| Size | FP32 traffic | Regime |
|---|---:|---|
| 48 × 64 | 49 KB | the source's example size: launch- and call-bound |
| 1024 × 1024 | 17 MB | L2-resident in every dtype |
| 2048 × 2048 | 67 MB (FP16: 34 MB) | the source's performance size: FP16 stays in L2, FP32 just exceeds it |
| 4096 × 4096 | 268 MB (FP16: 134 MB) | lab extension: DRAM-bound in every dtype |

## Reproducing the recorded results

[sweep.sh](../benchmarks/sweep.sh) runs everything behind [results](results.md): the test
suite as a gate, three benchmark runs, the 23 Nsight Compute reports and the chart.

```bash
PYTHON=.venv/bin/python TORCH_CUDA_ARCH_LIST=12.0 \
  problems/rgb_to_grayscale/benchmarks/sweep.sh artifacts/rgb_to_grayscale 960
python -m problems.rgb_to_grayscale.benchmarks.plot artifacts/rgb_to_grayscale/device_*.json
```

The benchmark runs take a few minutes (most of it compiling `pytorch_compile` for 12 shapes
and dtypes per run); the Nsight reports about ten more. The chart step needs matplotlib
and is skipped when it is missing.
