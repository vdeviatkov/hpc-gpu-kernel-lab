# Results

[← RGB to Grayscale](../README.md) · [Design](design.md) · **Results** ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

**Setup:** NVIDIA RTX 5080 (64 MiB L2, DRAM peak 960 GB/s), CUDA 13.2, PyTorch 2.14.1,
stock clocks, idle GPU. Each number is the median of 100 samples after 25 warmups, taken
as the median of three full runs (spread between runs ≤ 0.7% at 4096², ≤ 1.2 µs at 2048²).
One sample is one Python call timed with CUDA events, including output allocation.
Integer inputs 0–255. Measured from commit `210fbb8` plus the static-shape
`pytorch_compile` change. Commands are in [benchmarking](benchmarking.md).

## DRAM-bound: 4096 × 4096

Latency in µs, lower is better; effective bandwidth (`4·P·s` per second) in brackets.

| 4096 × 4096 | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| copy of the image (moves 1.5× the bytes) | 491.0 (820 GB/s) | 244.3 (824) | 244.1 (825) |
| `pytorch` (reference) | 1410.9 (190) | 1878.1 (71) | 1878.0 (71) |
| `pytorch_matmul` | 567.9 (473) | 452.9 (296) | 454.1 (296) |
| `pytorch_compile` | 321.8 (834) | 165.9 (809) | 166.0 (809) |
| `cuda_pixel` | 315.2 (852) | 159.3 (842) | 159.3 (842) |
| `cuda_vec` | 318.0 (844) | 161.7 (830) | 161.5 (831) |
| `cuda_smem` | **314.8 (853)** | **155.6 (863)** | **155.0 (866)** |
| `cuda_planar` | 809.4 (332) | 395.8 (339) | 395.8 (339) |

- `cuda_pixel`, `cuda_vec`, `cuda_smem` and `pytorch_compile` are within 2% (FP32) and
  7% (FP16) of each other, at 84–90% of the DRAM peak, and faster than a copy, which moves
  more bytes.
- In the profiler their FP32 kernels take the same time and the FP16 order changes
  ([below](#1-bytes-every-fused-kernel-is-at-the-lower-bound)): the differences here are
  per-call overhead and cache state, not kernel design.
- The eager reference is 4.5× (FP32) to 12× (FP16) slower; `cuda_planar` 2.6×.

![Effective bandwidth of every backend at 2048 × 2048 and 4096 × 4096 on RTX 5080. At 4096 × 4096 the CUDA kernels and torch.compile reach 809–866 GB/s, below the 960 GB/s DRAM peak; at 2048 × 2048 FP16 cuda_vec reaches 2060 GB/s because the image stays in L2.](../figures/rgb_bandwidth.svg)

## The source's size: 2048 × 2048

| 2048 × 2048 | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| `pytorch` (reference) | 255.5 | 320.6 | 320.4 |
| `pytorch_matmul` | 150.9 | 113.3 | 119.5 |
| `pytorch_compile` | 76.3 | 21.2 | 21.2 |
| `cuda_pixel` | 70.3 | 21.4 | 21.4 |
| `cuda_vec` | 77.5 | **16.3** | **16.1** |
| `cuda_smem` | **68.4** | 27.4 | 26.8 |
| `cuda_planar` | 155.0 | 35.3 | 35.3 |

- **FP32** (67 MB with the output, about the L2's size) still streams from DRAM: the CUDA
  kernels run at 866–982 GB/s, close to the peak, within 13% of each other.
- **FP16** (34 MB) stays in L2 between calls, and the variants separate: `cuda_vec` is the
  fastest at 2060 GB/s effective. These times include about 11 µs of fixed call cost
  (19.5 µs for `pytorch_compile`); the kernel-only times are in the next section.

## Why the variants differ

Nsight Compute, one launch per report ([how to capture and read](profiling.md)). ncu
locks clocks and times the kernel alone, so its durations differ from the tables above;
compare the kernels with each other.

### 1. Bytes: every fused kernel is at the lower bound

| 4096² FP32, caches flushed | Time | DRAM read | L2 read sectors | DRAM busy |
|---|---:|---:|---:|---:|
| *lower bound (`3·P·s` read; no kernel can read less)* | — | *201.3 MB* | *6.29 M* | — |
| `cuda_pixel` | 262.1 µs | 201.3 MB | 6.29 M | 93.9% |
| `cuda_vec` | 263.8 µs | 201.4 MB | 6.29 M | 93.8% |
| `cuda_smem` | 264.4 µs | 201.3 MB | 6.29 M | 94.0% |
| `pytorch_compile` | 264.5 µs | 201.3 MB | 6.29 M | 93.9% |
| `cuda_planar`: planar kernel only | 264.2 µs | 201.3 MB | 6.29 M | 94.0% |
| `cuda_planar`: + `deinterleave_rgb` | + 438.2 µs | + 201.3 MB | + 6.29 M | 88.1% |
| `pytorch` (5 kernels) | 1162 µs | **872 MB** | 27.3 M | ~93% |
| `pytorch_matmul` | 579.0 µs | 201.4 MB | 6.29 M | **46.6%** |

The italic row is computed, not measured. In FP16 the picture is the same: the fused kernels
read exactly 100.7 MB at 93–96% DRAM throughput, in 114–133 µs. The FP16 spread there
follows how much output each launch happened to write back during its run (2.9–16.7 MB),
and the order changes between ncu and the benchmark, so it is not a property of the kernels.

- **The stride-3 access costs nothing at the DRAM level.** `cuda_pixel`'s loads touch 12
  sectors per request instead of 4, but the second and third channel loads hit the sectors
  the first one brought into L1 (66.7% L1 hit rate), so L2 sees exactly the lower bound.
  `cuda_vec` (32 sectors per request, 50% L1 hit) and `pytorch_compile` (24, 83%) rely on
  the same reuse; only `cuda_smem` asks for exactly the bytes it needs (16 sectors per
  512-byte request, no reuse needed), and gains nothing from it.
- **The eager reference reads the image 3 times.** Each `0.299 * rgb[..., 0]` reads one
  channel through a strided view, which touches every sector of the image: three 201 MB
  reads, plus two adds over 134 MB of intermediates, 4.3× the lower bound. In FP16 it also
  converts the whole image to FP32 first, so it falls further behind.
- **`pytorch_matmul` is not memory-bound.** It reads the minimum, but cuBLAS's GEMV kernel
  reduces over an inner dimension of 3: half-empty or 3-lane warps (Source page), 19
  instructions per pixel against 0.3–2.5 for the others, and issue slots 62% (FP32) to 80%
  (FP16) busy. DRAM is busy only 47% (FP32) and 28% (FP16) of the time.
- **A planar layout would not help.** The planar kernel alone (perfectly contiguous loads)
  takes the same 264 µs as the interleaved kernels, so even data that is already planar
  gains nothing here; converting costs another 438 µs.

### 2. When the image is in L2: instructions and L2 bandwidth

| 2048² FP16, image in L2 | Kernel time | L2 throughput | Issue slots busy | Instructions / pixel | Top stall |
|---|---:|---:|---:|---:|---|
| `cuda_vec` | **7.7 µs** | **90.8%** | 25% | 0.32 | long scoreboard |
| `pytorch_compile` | 8.2 µs | 84.8% | 65% | 0.91 | long scoreboard |
| `cuda_planar`: planar kernel only | 12.9 µs (+31.0 convert) | 55.1% | 39% | 0.94 | long scoreboard |
| `cuda_pixel` | 14.0 µs | 49.2% | 45% | 1.19 | long scoreboard |
| `cuda_smem` | 20.2 µs | 34.0% | 53% | 2.07 | **barrier** |

All five hit L2 for 100% of sectors and leave DRAM idle (2–3%). With DRAM out of the way,
the limit becomes L2 bandwidth, and only kernels that issue few, wide requests reach it.
`cuda_vec` moves 33.5 MB in 7.7 µs (4.4 TB/s) with three 16-byte loads and one 16-byte
store per 8 pixels. `cuda_pixel` needs three 2-byte loads per pixel and reaches half the L2
bandwidth. `cuda_smem` executes 6.5× the instructions of `cuda_vec` (global → shared copy,
barrier, three shared loads), waits at `__syncthreads()`, and its FP16 shared loads have
2-way bank conflicts (about one extra cycle per load: 0.41 M conflicts over 0.39 M loads).
In FP32 at this size, every kernel is again 36–39 µs at 92–94% DRAM busy: 67 MB does not
fit beside other data in the 64 MiB L2.

### 3. Occupancy and registers

| 4096², caches flushed | `cuda_pixel` | `cuda_vec` | `cuda_smem` | `pytorch_compile` |
|---|---:|---:|---:|---:|
| Registers per thread (FP32 / FP16) | 16 / 16 | 40 / 25 | 36 / 34 | 16 / 17 |
| Warps active (FP32) | 74% | 83% | 93% | 85% |
| Issue slots busy (FP32) | 7.7% | 2.3% | 18.4% | 5.8% |

None of these limits a DRAM-bound launch: all four keep DRAM 94% busy.

## Small images

At 48 × 64 every backend measures its call cost: 10.8–11.1 µs for `cuda_pixel`, `cuda_vec`
and `cuda_smem`, 13.1 µs for `cuda_planar` (two launches and an allocation), 15–17 µs for
`pytorch_matmul`, 19.6 µs for `pytorch_compile` (guard checks before each call), 20.8 µs
(FP32) to 26.8 µs (FP16) for the eager reference, and 6.0 µs for the copy reference (one
copy, no Python dispatch). At 1024 × 1024 FP16, `cuda_vec` (11.0 µs) is the only kernel at
call cost; `cuda_pixel` takes 13.2 µs, `cuda_smem` 15.1 µs.

## Decisions

- **`cuda_vec` is the recommended kernel.** At DRAM-bound sizes it ties with the others
  (within 1% of the best in the profiler); with the image in L2 it is the fastest, at 91% of
  L2 bandwidth with the fewest instructions. It needs a 16-byte aligned image and falls
  back to `cuda_pixel` otherwise. `cuda_pixel` is the simplest kernel that reaches the DRAM
  limit.
- **The three-channel stride is not a DRAM problem.** L1 absorbs it, so DRAM and L2
  traffic stay at the lower bound; it costs instructions and L1/L2 throughput, which shows
  only when the data is cache-resident.
- **`cuda_smem` is not worth its complexity:** it fixes a coalescing "problem" that L1
  already solves and adds a barrier, 2–8× the instructions and FP16 bank conflicts. Its
  slight lead in the DRAM-bound benchmark (≤ 4%) does not appear in the profiler.
- **Converting to planar costs more than it saves:** 2.6× slower, and the planar kernel by
  itself is no faster than the interleaved ones at DRAM-bound sizes.
- **`torch.compile` is the honest PyTorch baseline:** its kernel matches ours at 4096² and is
  within 7% of `cuda_vec` from L2; what remains is its 19.5 µs call cost. It must be
  compiled per shape (or carefully), since its dynamic-shape kernel can be 6× slower
  depending on the first image it saw ([design](design.md#implementations)). The eager
  reference is the correctness oracle only; `image @ w` is a poor substitute for a fused
  kernel.

## Validation

- Test suite on the RTX 5080: **529 passed**; 86 skipped are the Triton backend, not yet
  implemented ([testing](testing.md)).
- Compute Sanitizer: memcheck **0 errors** in 344 CUDA tests (including misaligned images
  that take the fallback); racecheck **0 hazards**, synccheck and initcheck **0 errors** on
  the 172 `cuda_vec` and `cuda_smem` tests.

## Limitations

- One desktop GPU with unlocked clocks; buffers are reused between samples.
- The benchmark times whole Python calls; kernel-only comparisons come from single ncu
  launches, which run at base clocks.
- Triton is not implemented yet.

[sweep.sh](../benchmarks/sweep.sh) regenerates the raw JSON and the Nsight reports.
