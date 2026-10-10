# Results

[← Reverse Array](../README.md) · [Design](design.md) · **Results** ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

**Setup:** NVIDIA RTX 5080 (64 MiB L2, DRAM peak 960 GB/s), CUDA 13.2, PyTorch 2.14.1,
Triton 3.8.0, stock clocks, idle GPU. Each number is the median of 100 samples after 25
warmups, taken as the median of three full runs. Timing uses CUDA events around one call.
Inputs are uniform in [−1000, 1000]. Measured from commit `f0408a0` plus the Triton
backends. Commands are in [benchmarking](benchmarking.md).

## DRAM-bound: N = 100,000,000

Latency in µs, lower is better. "Aligned" N is a multiple of 16; "misaligned" is N + 1.
BF16 matches FP16 within 1% (except `cuda_pair` and `cuda_vec`, which are noisy).

| Backend | FP32 aligned | FP32 misaligned | FP16 aligned | FP16 misaligned |
|---|---:|---:|---:|---:|
| device copy (target) | 977.4 | 977.4 | 489.8 | 490.7 |
| `pytorch` | 1925.3 | 1927.6 | 951.7 | 951.7 |
| `cuda_pair` | 997.8 | 1001.9 | 671.0 | 540.2 |
| `cuda_vec` | 985.5 | 1003.7 (runs `pair`) | 492.8 | 586.8 (runs `pair`) |
| `cuda_tile` | **981.5** | **986.4** | 492.1 | **494.1** |
| `cuda_tile_async` | 989.4 | 990.8 | 495.1 | 495.2 |
| `triton_pair` | 992.8 | 995.7 | 495.0 | 499.1 |
| `triton_flip` | 986.6 | 988.6 | **492.0** | **494.1** |

- `cuda_tile` and `triton_flip` stay within 1.2% of a copy for every dtype and N.
- `cuda_vec` equals them only for aligned N; otherwise it falls back to `cuda_pair`.
- `cuda_pair` is the weak spot in FP16: 10–37% slower than a copy.
- PyTorch takes twice as long as a copy, as the [model](design.md#performance-model)
  predicts for `torch.flip` + `copy_`.
- Run-to-run spread is ≤ 0.4% for every backend except `cuda_pair` and `cuda_vec`
  (4.9% and 6.6%, both from the FP16 pair path).

At the source's performance size, N = 25,000,000 FP32 (also DRAM-bound): copy 244.1 µs,
`cuda_tile` 247.4, `cuda_vec` 247.6, `triton_flip` 249.3, `triton_pair` 251.5,
`cuda_pair` 258.5, PyTorch 457.0.

## Why the variants differ

Nsight Compute, one launch per report, at N = 100M ([how to capture and read](profiling.md)).
ncu locks clocks and flushes caches, so its durations differ from the table above; the
comparisons between variants are what matter. The analysis asks, in order, the four ways
a memory-bound kernel can lose to a copy: moving more bytes than needed, not keeping
DRAM busy, saturating another unit, or overhead.

### 1. Wasted bytes: is descending access coalesced?

A warp's accesses are grouped into 32-byte sectors. The ideal is 4 sectors per warp
request for FP32 (32 × 4 B) and 2 for FP16.

| `cuda_pair` | Load sectors/request | Store sectors/request | DRAM busy |
|---|---:|---:|---:|
| FP32, aligned N | 4.00 | 4.00 | 85.2% |
| FP32, misaligned N | 4.50 | 4.50 | 85.2% |
| FP16, aligned N | 2.00 | 2.00 | 74.1% |
| FP16, misaligned N | 2.50 | 2.50 | 66.6% |

Descending order costs nothing: aligned N reaches the ideal exactly. With misaligned N the
mirrored warp straddles one extra sector, raising the average over both halves by exactly
0.5. In FP32 that extra sector is usually fetched by the neighbouring warp anyway (an L2
hit), so time barely changes; in FP16 it is a 25% overhead per request.

### 2. Is DRAM kept busy?

| Report | DRAM busy | Instructions per element |
|---|---:|---:|
| `cuda_pair` FP32 | 85.2% | 0.45 |
| `cuda_pair` FP16 | **74.1%** | 0.45 |
| `cuda_vec` FP16 | 87.4% | 0.08 |
| `cuda_tile` FP16 | 87.6% | 0.92 |
| `cuda_tile` FP32 | 87.0% | 1.83 |

`cuda_pair` in FP16 leaves DRAM a quarter idle although 94% of warps are resident and
mostly waiting on memory: each 2-byte request carries too little data to keep enough
bytes in flight. 16-byte accesses fix that. `cuda_tile` executes 2–4× more instructions
than `cuda_pair` and is still faster: when a kernel waits on memory, instruction count
barely matters; request size does.

### 3. Does shared memory become a bottleneck?

| `cuda_tile` | Shared load conflicts | Shared store conflicts | Shared pipeline busy |
|---|---:|---:|---:|
| FP32, aligned N | 9.96 M | 6.42 M | 13.2% |
| FP16, aligned N | 9.88 M | 5.76 M | 24.3% |
| FP16, misaligned N | 9.98 M | 5.74 M | 23.7% |

The reversed reads are not pack-aligned, so bank conflicts occur, but the shared-memory
pipeline is 13–24% busy: it waits on DRAM like everything else, and the conflicts cost no
time. The memory units overall are similarly idle ("Mem Busy" ~18%) while the DRAM link
is ~87% utilized ("Max Bandwidth").

### 4. Where does `cuda_tile_async` lose?

| FP32, aligned N | `cuda_tile` | `cuda_tile_async` |
|---|---:|---:|
| Kernel duration (ncu) | 923.0 µs | 927.7 µs |
| Instructions per element | 1.83 | 1.96 |
| Main stalls (per issued instruction) | memory 32, barrier 12 | memory 22, barrier 7, MIO throttle 7 |
| Shared store conflicts | 6.4 M | 0 |
| Blocks per SM limited by registers / shared memory | 6 / 11 | 6 / 10 |

`cp.async` works as designed: shared-store conflicts disappear and fewer warps wait on
memory. It adds 7% more instructions and a new stall, MIO throttle (the queue that
issues the async copies), and there is no computation to hide them behind. Occupancy is
unchanged: registers limit both kernels to 6 blocks per SM.

### Unexplained: `cuda_pair` FP16 is slower with aligned N

In the benchmark, `cuda_pair` FP16 takes 671 µs for N = 100,000,000 and 540 µs for
N + 1, consistently across runs; a single profiled launch shows the opposite (535 µs
aligned, 597 µs misaligned). The cause is not established. A plausible but unverified
hypothesis is that, with aligned N, each element and its mirror map to DRAM channels in a
pattern that concentrates traffic in steady state. It does not affect the conclusions:
the 16-byte kernels avoid it.

## Triton

| Kernel | Aligned N | Misaligned N |
|---|---|---|
| `triton_pair` | front half 128-bit; back half scalar, 4 (FP32) or 8 (FP16) loads per thread | all scalar |
| `triton_flip` | all 128-bit, reversal through shared memory | all scalar |

Triton specializes on whether N is divisible by 16, so it vectorizes only when it can
prove alignment, and never for descending addresses. Each thread owns 4 (FP32) or 8
(FP16) consecutive elements; when those are loaded with scalar instructions, every
instruction touches 4× (FP32) or 8× (FP16) the sectors it uses, and the repeats hit in
L1. At DRAM-bound sizes that costs almost nothing: `triton_flip` matches `cuda_tile`
within 0.5%, `triton_pair` within 1.2%. Where data comes from L2 it does cost: at
N = 25M FP16 (50 MB, cache-resident in place) misaligned Triton takes 98–111 µs against
48–52 µs for the CUDA kernels.

## Small inputs

At N = 1025 every backend is dominated by fixed per-call overhead: 9.2–9.3 µs for PyTorch
and CUDA, 10.8–11.1 µs for Triton, against 5.9 µs for a copy. At N = 1M (4 MB, L2-resident)
the tile kernels cost about 2 µs more than `cuda_pair` (11.2 vs 9.3 µs FP32): their
load → barrier → store sequence is visible when there is little data to overlap it with.

## Validation

- Test suite on the RTX 5080: **868 passed**, none skipped ([testing](testing.md)).
- Compute Sanitizer: memcheck 0 errors for all CUDA and Triton backends; racecheck 0
  shared-memory hazards for `cuda_tile`, `cuda_tile_async` and both Triton kernels;
  synccheck 0 errors for `cuda_tile_async`.

## Limitations

- One desktop GPU with unlocked clocks; buffers are reused between samples.
- In place, 25M FP16/BF16 stays in L2 between calls; use the 100M rows for DRAM behavior.
- The `cuda_pair` FP16 aligned-vs-misaligned inversion above is unexplained.
- `cuda_vec` falls back to `cuda_pair` for misaligned N; falling back to `cuda_tile`
  would remove its FP16 penalty.

Raw JSON and the Nsight reports are in `results/rtx5080/` (not tracked by Git);
[sweep.sh](../benchmarks/sweep.sh) regenerates both.
