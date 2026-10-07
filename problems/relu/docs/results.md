# Results

[← ReLU](../README.md) · [Design](design.md) · **Results** ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

**Setup:** NVIDIA RTX 5080 (64 MiB L2, DRAM peak 960 GB/s), CUDA 13.2, PyTorch 2.14.1,
Triton 3.8.0, stock clocks. Each number is the median of 100 samples after 25 warmups,
taken as the median of three full runs (spread between runs ≤ 1.3%). Timing uses CUDA
events around one call, including output allocation. Inputs are uniform in [−100, 100].
Commands are in [benchmarking](benchmarking.md).

## Large vectors: N = 25,000,000

Latency in µs, lower is better.

| Backend | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| device copy (target) | 244.2 | 121.7 | 122.2 |
| `pytorch` | 245.2 | 122.3 | 122.4 |
| `cuda_select` | 246.4 | 140.8 | 140.8 |
| `cuda_branch` | 247.3 | 144.8 | 144.8 |
| `cuda_fmax` | 246.3 | 140.7 | 139.2 |
| `cuda_vec` | 246.4 | 125.9 | 125.5 |
| `triton` | 250.0 | 127.3 | 127.4 |

- **FP32:** every backend is within 2.4% of a copy; the formula and load width do not
  matter.
- **FP16/BF16:** the scalar kernels are 15–19% slower than a copy; `cuda_vec` and
  Triton are 3–5% slower; PyTorch matches it.

## Why the variants differ

Nsight Compute counts for one launch at N = 25M, mixed signs
([how to capture](profiling.md)).

| Backend | Instructions per element (FP32 / FP16) | Stores per warp (FP32 / FP16) | DRAM busy, FP16 | Divergent branches |
|---|---:|---:|---:|---:|
| `cuda_select` | 0.63 / 0.69 | 1 / 1 | 67% | 0 |
| `cuda_branch` | 0.75 / 0.78 | **2 / 2** | 64% | 0 |
| `cuda_fmax` | 0.59 / 0.66 | 1 / 1 | 67% | 0 |
| `cuda_vec` | 0.26 / 0.21 | 0.25 / 0.125 | 93% | 1 (tail) |
| `triton` | 0.18 / 0.16 | 0.25 / 0.125 | 93% | 0 |
| `pytorch` | 0.20 / 0.28 | 0.25 / 0.25 | 93% | 0 FP32; see note |

- **No branches.** In the SASS, `cuda_select` is `FSETP` + `FSEL`, `cuda_fmax` a single
  `FMNMX`, and `cuda_branch` predicated stores: negative lanes store zero and exit, the
  others store `x`. None contains a data-dependent jump.
- **The `if/else` doubles stores in mixed warps.** A warp with both signs issues both
  predicated stores, each covering part of the warp.
- **2-byte accesses starve DRAM.** Scalar FP16 kernels keep DRAM only ~67% busy; 16-byte
  accesses (one store per 128 FP32 or 256 FP16 elements) reach 93%. Triton chooses
  16-byte accesses on its own.
- **FP32 is already saturated.** Scalar 4-byte accesses keep DRAM 89% busy, the same as
  every vectorized variant, so the extra instructions cost nothing.

Note: PyTorch's FP16 kernel shows data-dependent divergent branches (97,656 with mixed
signs, 0 with one sign) without a measurable time effect; it is not our code.

Nsight measures single launches with caches controlled differently from normal runs, so
its durations are not comparable with the table above; use it for counts and ratios.

## Sign patterns

Same N = 25M, FP16, latency in µs. `alternating` flips the sign every element, so every
warp mixes signs; `warp_blocks` gives each warp a single sign.

| Backend | mixed | alternating | warp_blocks | positive | negative |
|---|---:|---:|---:|---:|---:|
| `cuda_select` | 140.8 | 140.8 | 140.8 | 140.7 | 140.2 |
| `cuda_branch` | 144.8 | 144.9 | 142.9 | 142.9 | 140.9 |
| `cuda_vec` | 125.9 | 124.5 | 125.5 | 124.5 | 125.3 |

Only `cuda_branch` depends on the data: mixed warps cost 1.3% more than single-sign
warps, and all-negative warps are fastest because they exit after one store. In FP32 the
same effect is under 1%. The branch-free variants do not change.

## Small inputs

At N = 1025 and N = 1M, every call is dominated by fixed overhead: about 6.6–6.9 µs for
PyTorch, 10–12 µs for the CUDA kernels and 12.3 µs for Triton. The kernels themselves
take under 2 µs (1.8 µs for `cuda_select`, 1.9 µs for PyTorch at N = 1025). The gap is
host time: our CUDA call path spends 7.1 µs of CPU time per call against 3.9 µs for
PyTorch, of which the extension call itself is 1.75 µs; the rest is Python dispatch,
output allocation and the extension lookup.

## Triton

Tile and warp sweep at N = 25M (µs, median of three runs of 50 samples):

| Elements / warps | 256/4 | 256/8 | 512/4 | 1024/4 (default) | 2048/4 | 4096/4 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 249.4 | 248.3 | 249.5 | 250.4 | 252.4 | 253.5 |
| FP16 | 126.4 | **142.8** | 126.4 | 127.5 | 128.6 | 128.6 |

All configurations are within 2% except 256 elements with 8 warps in FP16: that gives
each thread a single 2-byte element, reproducing the scalar-access slowdown.

## Validation

- Test suite on the RTX 5080: **361 passed**, none skipped. Compute Sanitizer memcheck:
  **0 errors** in all CUDA and Triton tests ([testing](testing.md)).
- Every backend except `cuda_fmax` keeps −0.0 and NaN like `torch.relu`; `cuda_fmax`
  returns +0.0 for both (checked by hand; special values are not in the test suite).

## Limitations

- One desktop GPU with unlocked clocks; buffers are reused between samples.
- Special values (NaN, ±inf, −0.0) are not part of the automated tests.
- Timing includes output allocation and Python dispatch, which dominate below ~1M
  elements.
