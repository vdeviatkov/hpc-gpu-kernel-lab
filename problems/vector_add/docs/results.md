# Results

[← Vector Addition](../README.md) · [Design](design.md) · **Results** ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

## Measurement environment

Run date: 2026-10-02 UTC. RTX 5080 (SM120, 84 SMs, 64 MiB L2), driver 595.91.07,
CUDA toolkit 13.2.86, Python 3.12.3, PyTorch 2.14.1 (CUDA 13.0), Triton 3.8.0,
JAX/jaxlib 0.11.2. Stock clocks, desktop display active, one GPU; clocks not locked.
Each case: correctness check, 25 warmups, 100 samples. Buffers are reused without
cache flushing. Backend order is shuffled per case with a fixed seed.
Measured from the working tree of this revision (base commit `3001d91` plus the
uncommitted changes recorded in the JSON metadata). Commands: [benchmarking](benchmarking.md).

## N = 25,000,000 (DRAM-bound)

Median of three complete run medians; the range shows those three medians, not a
confidence interval. Scope: CUDA events around an eager call on the current stream.

| Backend | FP32 µs | FP32 GB/s | FP16 µs | FP16 GB/s | BF16 µs | BF16 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| `pytorch` | 359.26 (358.86–359.63) | 835.0 | 177.60 (177.57–177.66) | 844.6 | **175.55** (175.46–175.65) | 854.4 |
| `cuda_scalar` | **354.78** (354.59–355.60) | 845.6 | 183.81 (183.81–183.84) | 816.1 | 183.81 (183.81–183.87) | 816.1 |
| `cuda_grid_stride` | 364.03 (363.74–364.06) | 824.1 | 182.78 (182.75–182.78) | 820.6 | 180.77 (180.74–180.83) | 829.8 |
| `cuda_vec` | 358.64 (358.00–358.88) | 836.5 | **173.73** (173.63–174.32) | 863.4 | 175.68 (175.63–175.71) | 853.8 |
| `cuda_vec_grid_stride` | 359.78 (359.76–359.81) | 833.9 | 178.72 (178.66–178.75) | 839.3 | 179.49 (179.49–179.55) | 835.7 |
| `triton` | 363.95 (363.90–364.02) | 824.3 | 179.41 (179.39–179.55) | 836.1 | 181.84 (181.82–182.45) | 824.9 |
| `copy_reference` | 244.70 | 817.3 | 116.08 | 861.5 | 114.18 | 875.8 |

`copy_reference` GB/s counts two streams (`2*N*s`); all other rows count three.
See the [performance model](design.md#performance-model) for why copy is a reference,
not a ceiling.

## What the 2×2 design shows

Nsight Compute, one profiled launch per variant at N = 25M
([how to capture](profiling.md)). Profiler replay controls caches and clocks
differently from normal timing, so durations differ from the table above. The
ordering matches.

| Variant | Dtype | Duration µs | DRAM % of peak | Warps active % | Instructions | Registers |
|---|---|---:|---:|---:|---:|---:|
| `cuda_scalar` | FP32 | 302.7 | 91.8 | 82.6 | 17.19 M | 16 |
| `cuda_vec` | FP32 | 304.2 | 91.3 | 86.5 | 6.25 M | 18 |
| `cuda_grid_stride` | FP32 | 322.5 | 89.8 | 92.1 | 9.31 M | 34 |
| `cuda_vec_grid_stride` | FP32 | 313.0 | 90.7 | 61.9 | 6.14 M | 54 |
| `cuda_scalar` | FP16 | 156.4 | 88.5 | 80.7 | 19.53 M | 16 |
| `cuda_vec` | FP16 | 128.1 | 92.6 | 87.2 | 5.47 M | 26 |
| `cuda_grid_stride` | FP16 | 140.8 | 91.5 | 92.8 | 11.85 M | 29 |
| `cuda_vec_grid_stride` | FP16 | 128.4 | 93.5 | 76.0 | 6.79 M | 48 |

- **Vectorization, loop held fixed.** FP32: scalar → vec cuts instructions 64% but
  leaves DRAM throughput unchanged; latency moves +1.1% (direct) and −1.2% (grid-stride).
  At four bytes per element, scalar loads already keep DRAM about 92% busy. FP16: the
  same change cuts instructions 72% and is 5.5% faster (direct). Two-byte scalar accesses
  issue too many memory instructions to keep DRAM busy.
- **Grid-stride loop, access width held fixed.** FP32 scalar → grid-stride is 2.6%
  slower. The compiler unrolls the loop (34 registers vs 16) and the 4096-block cap is a
  fixed starting choice. The vector loop reaches 54 registers and 62% active warps.
  An earlier version of this study compared the vector loop against `cuda_scalar` and
  attributed the gap to vectorization; the 2×2 design shows it came from the loop.
- **Best per dtype.** FP32: `cuda_scalar`, 1.3% faster than PyTorch. FP16: `cuda_vec`,
  2.2% faster than PyTorch. BF16: `cuda_vec` and PyTorch tie (within 0.1%).

## Size sweep and small vectors

The [overview chart](../README.md) comes from 13 sizes per dtype
([plot script](../benchmarks/plot.py)). At N = 1025, device-scope medians are
8.3–8.5 µs for PyTorch and all CUDA variants, 13.7 µs for Triton, and 5.4 µs for
`copy_reference`. An earlier Nsight Systems capture of the scalar kernel at N = 1025
measured 0.61 µs median kernel duration and 1.92 µs median `cudaLaunchKernel` time,
so event timing at this size is mostly dispatch, not memory traffic. The L2 peak is
at 50 MB for FP32 (N = 4,194,304; 3.1 TB/s); at 101 MB, the working set no longer
fits and bandwidth drops to ~1 TB/s, then to the DRAM plateau.

## API scope (one run, FP32)

Host-observed allocation, dispatch and completion; JAX receives identical GPU inputs
through DLPack. Do not mix with the device-scope tables.

| Backend | N = 1025 µs | N = 25,000,000 µs |
|---|---:|---:|
| `pytorch` | 8.24 | 360.05 |
| `cuda_scalar` | 9.53 | 355.97 |
| `cuda_grid_stride` | 8.50 | 366.02 |
| `cuda_vec` | 8.52 | 359.46 |
| `cuda_vec_grid_stride` | 8.72 | 363.19 |
| `triton` | 13.58 | 366.00 |
| `jax` | 23.61 | 385.64 |

## Validation

- GPU test suite: **811 passed, 0 skipped** ([testing](testing.md)).
- Compute Sanitizer memcheck: **508 CUDA tests, 0 errors**. Covers values/tails, guard
  buffers, output alignment for every dtype, multi-iteration grid-stride loops and
  nondefault streams.
- A misaligned run (`--offset 1`) confirms that the vector variants take their scalar
  fallback and match its timing.
- CI runs lint, formatting and the CPU test suite on every push.

## Limitations

One desktop GPU, unlocked clocks, reused buffers, and one API-scope run. Multi-GPU
device selection is implemented (device guard) but not tested. Geometry sweeps
(threads, Triton tiles) were last run on the previous revision and showed effects
of about 1%; they are not repeated here. No universal fastest backend is claimed.

Follow-ups: rotating buffers around the L2 transition, an uncapped or
SM-count-derived grid for the grid-stride variants, and CUDA Graphs for small sizes.

## Raw data

Raw JSON, Nsight Compute metrics and resource usage are kept in
`results/rtx5080/current/` in the local workspace. That directory is ignored by Git;
the tables on this page are the tracked summary.
