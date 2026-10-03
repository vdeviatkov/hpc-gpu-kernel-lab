# Results

[← Vector Addition](../README.md) · [Design](design.md) · **Results** ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

**Setup:** NVIDIA RTX 5080 (64 MiB L2, DRAM peak 960 GB/s), CUDA 13.2, PyTorch 2.14.1,
Triton 3.8.0, JAX 0.11.2, stock clocks. Each number is the median of 100 samples after
25 warmups, taken as the median of three full runs (spread between runs ≤ 0.4%).
Timing uses CUDA events around one call. Commands are in [benchmarking](benchmarking.md).

## Large vectors: N = 25,000,000

Latency in µs, lower is better. **Bold** marks the fastest per dtype.

| Backend | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| `pytorch` | 359.3 | 177.6 | **175.6** |
| `cuda_scalar` | **354.8** | 183.8 | 183.8 |
| `cuda_grid_stride` | 364.0 | 182.8 | 180.8 |
| `cuda_vec` | 358.6 | **173.7** | 175.7 |
| `cuda_vec_grid_stride` | 359.8 | 178.7 | 179.5 |
| `triton` | 364.0 | 179.4 | 181.8 |

Every backend reaches 816–863 GB/s, which is 85–90% of the 960 GB/s peak. This size
is limited by memory bandwidth, so the backends differ by at most 6%.

## Why the variants differ

The four CUDA kernels differ in two choices (see [design](design.md#implementations)):
scalar or 16-byte vector loads, and one element per thread or a grid-stride loop.
Nsight Compute counts at N = 25M (instructions are counted per warp, i.e. per
32 threads, and divided by N):

| Kernel | Vector loads | Loop | Registers (FP32 / FP16) | Instructions per element (FP32 / FP16) |
|---|:---:|:---:|---:|---:|
| `cuda_scalar` | | | 16 / 16 | 0.69 / 0.78 |
| `cuda_vec` | ✓ | | 18 / 26 | 0.25 / 0.22 |
| `cuda_grid_stride` | | ✓ | 34 / 29 | 0.37 / 0.47 |
| `cuda_vec_grid_stride` | ✓ | ✓ | 54 / 48 | 0.25 / 0.27 |

Instruction count is not the bottleneck at this size: the kernels spend most of
their time waiting for memory.

- **Vector loads help FP16, not FP32.** For FP32, scalar loads already keep DRAM 92%
  busy, so cutting instructions by 64% saves nothing. For FP16, 2-byte scalar loads
  leave DRAM 88.5% busy; vector loads raise that to 92.6% and save 5.5% of the time.
- **The grid-stride loop costs time.** The compiler unrolls it, which raises register use
  and reduces the number of active warps. FP32 is 2.6% slower with the loop than without.

## Small vectors

At N = 1025 every call costs about the same regardless of the work: **8.3 µs** for
PyTorch and all CUDA kernels, **13.7 µs** for Triton. The kernel itself runs for only
0.6 µs (Nsight Systems); the rest is launch and dispatch overhead. Including allocation
and synchronization, JAX takes 23.6 µs.

Between these two extremes, the three buffers fit in the 64 MiB L2 cache and stay there,
because the benchmark reuses them on every call. Effective bandwidth then reaches about
3 TB/s, above the 960 GB/s DRAM peak, because the data never goes to DRAM
(see the [overview chart](../README.md)). This is a warm-cache best case.

## Validation

- GPU test suite: **811 passed**, none skipped. Compute Sanitizer memcheck: **0 errors**
  in 508 CUDA tests ([testing](testing.md)).
- Misaligned buffers make the vector kernels use their scalar fallback, as designed.

## Limitations

- One desktop GPU with unlocked clocks; buffers are reused between samples.
- Multi-GPU device selection is implemented but not tested.
- Block-size and Triton tile sweeps changed results by about 1% on the previous revision
  and were not repeated.
