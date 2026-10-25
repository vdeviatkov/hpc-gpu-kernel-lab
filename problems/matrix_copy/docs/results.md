# Results

[← Matrix Copy](../README.md) · [Design](design.md) · **Results** · [Testing](testing.md) ·
[Benchmarking](benchmarking.md) · [Profiling](profiling.md)

**Session:** 2026-10-23 PDT (2026-10-24 03:12 UTC), cuda.0, RTX 5080 (84 SMs, 64 MiB L2),
CUDA toolkit 13.4.92, PyTorch 2.14.1 (CUDA 13.0), Triton 3.8.0, driver 615.71.09,
Nsight Compute 2026.3.1. Clocks unlocked; no other compute processes. Every case passed
its correctness gate before timing (1436 tests passed first).

Tables show the **median of three run medians**, each from 100 CUDA-event samples after 25
warmups. Inputs are reused and output allocation is included. The run-median spread is
0.15% (median) at N ≥ 4096 and 0.8% at 1024–3072. The exceptions are cases next to the L2
boundary, called out below. The `environment.commit` in the JSON is the GPU host's stale
checkout, not the code that ran; `artifacts/matrix_copy/remote_source_sha256.txt` hashes
the sources that ran. Compact evidence: [measured_summary.json](../benchmarks/measured_summary.json).

![Effective bandwidth against logical traffic per call: every coalesced kernel reaches about 800 GB/s, just under memcpy, past the L2 boundary; below it memcpy leads, cuda_stream falls off early, and the column-major kernel stays near 470 (FP32) or 130 GB/s (FP16) throughout.](../figures/bandwidth_vs_size.svg)

## DRAM-bound: how close to memcpy

Latency in µs, effective GB/s in parentheses, and the copy reference's latency over the
backend's (`% of copy`, 100% = as fast as memcpy).

| Backend | 4096² FP32 (source) | % of copy | 8192² FP32 | % of copy | 8192² FP16 | % of copy |
|---|---:|---:|---:|---:|---:|---:|
| copy reference (memcpy) | 162.2 (828) | 100 | 655.4 (819) | 100 | 325.3 (825) | 100 |
| `pytorch` (`clone`) | 163.4 (821) | 99.3 | 656.3 (818) | 99.9 | 326.3 (823) | 99.7 |
| `cuda_flat` | 164.5 (816) | 98.6 | 656.8 (817) | 99.8 | 353.9 (759) | 91.9 |
| `cuda_2d` | 167.5 (801) | 96.8 | 652.9 (822) | 100.4 | 445.0 (603) | 73.1 |
| `cuda_vec` | 167.5 (801) | 96.8 | 657.0 (817) | 99.8 | 332.4 (808) | 97.9 |
| `cuda_vec_ilp2` | 169.2 (793) | 95.9 | 663.1 (810) | 98.8 | 333.2 (806) | 97.6 |
| `cuda_vec_ilp4` | 168.6 (796) | 96.2 | 665.2 (807) | 98.5 | 336.3 (798) | 96.7 |
| `cuda_vec_gridstride` | 171.7 (782) | 94.5 | 672.2 (799) | 97.5 | 337.6 (795) | 96.4 |
| `cuda_stream` | 169.5 (792) | 95.7 | 661.0 (812) | 99.2 | 335.5 (800) | 97.0 |
| `triton` | 171.6 (782) | 94.5 | 665.5 (807) | 98.5 | 332.8 (806) | 97.7 |
| `triton_2d` | 174.7 (768) | 92.8 | 671.1 (800) | 97.7 | 341.2 (787) | 95.3 |
| `cuda_2d_colmajor` | 286.3 (469) | 56.7 | 1305.0 (411) | 50.2 | 1864.5 (144) | 17.4 |

- **memcpy reaches 819–828 GB/s, 85–86% of the nominal 960 GB/s.** That is the practical
  ceiling; nothing here beats it outside noise.
- **FP32: every coalesced kernel is within 2.5% of memcpy at 8192²**, scalar ones included.
  At 4096² the spread is 1.4–7%, partly the fixed call cost below.
- **FP16 is where width matters:** 2-byte scalar `cuda_flat` drops to 91.9% of copy and
  `cuda_2d` to 73.1%. Every 16-byte variant and Triton stays within 2–5%.
- BF16 runs the same kernels as FP16 ([design](design.md#moving-bits-not-values)); its
  table agrees within noise and is in the summary JSON.

## The L2 cliff

Below the cliff, the matrix stays in L2 between calls, and the copy reaches 2.0–2.75 TB/s.
The cliff falls where the **resident footprint** reaches 64 MiB, not the per-call
traffic. In `lab.bench` the previous output is still alive when the next is allocated,
so the footprint is 3·N²·e. [steady_state.py](../benchmarks/steady_state.py) times the same
call when it drops the previous output first (one reused output block, 2·N²·e):

| Case | Footprint 3·N²·e / 2·N²·e | memcpy, alternating | memcpy, reused | `cuda_vec`, alternating | `cuda_vec`, reused |
|---|---:|---:|---:|---:|---:|
| FP32 2304² | 63.7 / 42.5 MB | 16.4 | 16.1 | 20.3 | 20.1 |
| FP32 2560² | 78.6 / 52.4 MB | 57.0 | 18.3 | 62.0 | 22.3 |
| FP32 2896² | 100.6 / 67.1 MB | 82.5 | 24.2 | 85.5 | 26.8 |
| FP32 3072² | 113.2 / 75.5 MB | 91.7 | 63.3 | 96.6 | 66.8 |
| FP32 3584² | 154.1 / 102.8 MB | 125.3 | 118.3 | 129.5 | 123.1 |
| FP16 4096² | 100.7 / 67.1 MB | 81.2 | 22.2 | 85.2 | 26.0 |
| FP16 4608² | 127.4 / 84.9 MB | 103.7 | 88.5 | 107.9 | 89.1 |

Latency in µs (single runs of 100 samples; the alternating column agrees with the
benchmark within 10%, closer away from the boundary). Both columns fall where their own footprint crosses 64 MiB (67.1 MB):
alternating between FP32 2304² and 2560², reused between 2896² and 3072². In FP16 the
boundary moves √2 further out, as expected. **The source's 4096² FP32 case moves 134 MB
and is DRAM-bound either way; 4096² FP16 is L2-resident only if the output is reused**:
3.3× faster (26.0 vs 85.2 µs).

Below the cliff, memcpy leads every kernel by about 4 µs at 2048² FP32 (12.2 vs 16.4 µs for
`cuda_vec`). The fixed cost of one call accounts for that gap: at 512², where the kernel
is negligible, memcpy takes 6.1 µs, the CUDA extension about 10.3 µs and Triton about
14.2 µs. Kernel-only times are in the next section.

## Why the variants differ

Nsight Compute, one launch per report, at ncu's locked base clocks: compare the kernels
with each other, not with the benchmark. Cold: 4096², caches flushed. Warm: 2048², the
matrix in L2 after the warmups. Sectors/request are for loads; stores are identical.

| 4096² FP32, cold | Time µs | DRAM read MB | DRAM write MB | DRAM busy % | L2 busy % | Sectors/req | Warps active % | Regs | Warp instr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *minimum (N²·e)* | | *67.1* | *≤ 67.1* | | | | | | |
| `cuda_flat` | 112.9 | 67.2 | 30.0 | 90.9 | 29.7 | 4 | 79.7 | 16 | 9.44 M |
| `cuda_2d` | 140.4 | 67.2 | 42.2 | 82.3 | 24.6 | 4 | 76.2 | 32 | 50.9 M |
| `cuda_vec` | 116.4 | 71.1 | 27.0 | 89.0 | 26.9 | 16 | 82.8 | 18 | 3.54 M |
| `cuda_vec_ilp2` | 109.2 | 67.1 | 26.8 | 90.9 | 28.9 | 16 | 84.2 | 26 | 2.62 M |
| `cuda_vec_ilp4` | 113.7 | 67.1 | 27.7 | 88.2 | 27.5 | 16 | 79.9 | 40 | 2.00 M |
| `cuda_vec_gridstride` | 115.8 | 67.1 | 31.6 | 90.1 | 27.0 | 16 | 71.2 | 36 | 1.11 M |
| `cuda_stream` | 110.5 | 67.2 | 27.8 | 90.8 | 28.0 | 16 | 82.7 | 18 | 3.54 M |
| `triton` | 109.7 | 67.2 | 26.7 | 90.5 | 28.9 | 16 | 86.3 | 18 | 1.90 M |
| `triton_2d` | 120.7 | 71.3 | 27.4 | 86.4 | 26.1 | 16 | 63.7 | 56 | 1.95 M |
| `cuda_2d_colmajor` | 283.6 | 77.0 | 45.0 | 45.4 | 87.0 | **32** | 86.9 | 30 | 46.1 M |

| 4096² FP16, cold | Time µs | DRAM busy % | L2 busy % | Sectors/req | Warp instr. |
|---|---:|---:|---:|---:|---:|
| `cuda_flat` | 75.7 | 60.2 | 23.8 | 2 | 9.44 M |
| `cuda_2d` | 105.0 | 45.0 | 19.7 | 2 | 50.9 M |
| `cuda_vec` | 39.3 | 94.5 | 39.3 | 16 | 1.77 M |
| `cuda_vec_ilp2` / `ilp4` / `gridstride` | 39.2 / 41.0 / 39.6 | 94.2–94.6 | 38.7–40.7 | 16 | 1.57 / 1.26 / 0.70 M |
| `cuda_stream` | 38.8 | 94.3 | 39.9 | 16 | 1.77 M |
| `triton` / `triton_2d` | 39.4 / 43.1 | 94.6 / 87.1 | 39.2 / 37.4 | 16 | 1.38 / 1.16 M |
| `cuda_2d_colmajor` | 465.4 | 11.1 | **99.4** | 32 | 46.1 M |

| 2048², warm: kernel µs (L2 busy %) | FP32 | FP16 |
|---|---:|---:|
| `cuda_flat` | 12.4 (67.2) | 11.9 (37.8) |
| `cuda_2d` | 22.5 (38.1) | 22.5 (23.0) |
| `cuda_vec` | 9.5 (83.5) | 5.4 (73.7) |
| `cuda_vec_ilp2` | 9.5 (83.1) | 5.6 (71.6) |
| `cuda_vec_ilp4` | 10.0 (78.9) | 6.0 (64.6) |
| `cuda_vec_gridstride` | 9.7 (81.1) | 5.9 (67.1) |
| `cuda_stream` | 9.5 (83.5) | 5.6 (71.3) |
| `triton` | 15.8 (49.5) | 5.5 (72.7) |
| `triton_2d` | 10.2 (76.3) | 5.7 (72.5) |
| `cuda_2d_colmajor` | 64.1 (96.3) | 117.8 (98.3) |

All warm captures hit L2 for 89–100% of sectors with DRAM idle.

### 1. Every coalesced kernel reads the minimum

Coalesced kernels read 67.1–71.3 MB from DRAM cold, the minimum N²·e plus at most 6%.
They write only 27–42 MB *during* the kernel: the rest of the output is still dirty in L2
when the kernel ends and is written back afterwards. That is also why cold ncu times
(110–120 µs for 134 MB of logical traffic) look faster than 960 GB/s allows: DRAM moves
about 97 MB inside the timed window. At ~90% DRAM busy, nothing is left to optimize in the
cold FP32 case.

### 2. The uncoalesced kernel is L2-throughput-bound, not DRAM-bound

`cuda_2d_colmajor` touches 32 sectors per warp request instead of 4. Its block's 8 warps
cover 8 adjacent columns, so for FP32 the 8 touches of each 32-byte sector come from the
same block and L2 serves 7 of them (86% hit rate cold). DRAM reads only 15% more than the
minimum (77.0 MB), but L2 runs at 87% of its throughput and DRAM at 45%: **2.0× slower than
memcpy at 8192² FP32**. In FP16 the block covers only 16 bytes of each sector, so half of
every sector moved is wasted and L2 saturates (99.4%): **5.7× slower than memcpy (17.4%)**.
In the L2-resident regime it stays at the same ~470 / ~130 GB/s, so its slowdown grows to
5.9× (FP32 2048²) and 19× (FP16 2896²) against memcpy. Caches hide the extra DRAM traffic
but not the extra L1/L2 transactions.

### 3. Vectorization matters for 2-byte elements, and once data is in L2

- **FP32, DRAM-bound: no.** 4-byte scalar warps already issue 128-byte requests (4 sectors),
  and `cuda_flat` keeps DRAM 91% busy with 2.7× the instructions of `cuda_vec`.
- **FP16, DRAM-bound: yes, 1.9× at kernel level.** A 2-byte warp request moves only 64 bytes.
  `cuda_flat` reaches 60% DRAM busy in 75.7 µs; 16-byte packs reach 94.5% in 39.3 µs, with
  5.3× fewer instructions. In the benchmark the gap is smaller (91.9% vs 97.9% of copy at
  8192²) because call overhead and dirty write-back are shared.
- **In L2, both widths: yes.** With DRAM out of the way the limit becomes L2 bandwidth and
  request issue: warm `cuda_vec` takes 9.5 µs FP32 (83.5% L2 busy) against 12.4 µs for
  `cuda_flat` (67.2%), and 5.4 µs against 11.9 µs in FP16.

### 4. ILP and fewer blocks do not help

At full occupancy, one 16-byte pack per thread already puts up to 84 SMs × 48 warps × 512
bytes ≈ 2 MB in flight. At 960 GB/s that covers about 2 µs of latency, more than a loaded
DRAM round trip should take (an estimate; latency was not measured). More packs per thread
only queue:
the LG-throttle stall grows from ~0 (`cuda_vec`) to 146 (`ilp4`) and 368
(`gridstride`) cycles per issued instruction (cold FP32), while time stays within ±4%. In
the benchmark ILP2/ILP4/grid-stride are 0.2–2.3% *slower* than `cuda_vec` at 8192². The
grid-stride kernel launches one wave (504 blocks) and is also unrolled 4× by nvcc
([SASS](#sass)), so it does not isolate "fewer blocks" cleanly.

### 5. Streaming cache hints hurt below L2 and do not help above it

`__ldcs`/`__stcs` mark lines evict-first, which throws away exactly the reuse between calls
that makes small sizes fast: at 2304² FP32, `cuda_stream` takes 53.1 µs against 23.2 µs for
`cuda_vec` (2.3×), and it falls off the cliff at the smallest size of any kernel (see the
figure). Above L2 there is nothing to protect, and it is 0.6% slower than `cuda_vec`
at 8192² FP32 (661.0 vs 657.0 µs), within 1% in FP16. Inside one kernel launch the hints
stay within the spread of the other pack variants (cold FP32 110.5 µs against 109.2–116.4,
warm equal to `cuda_vec`). Note that `__ldcs`
also drops the read-only `.CONSTANT` load path, so it changes two things at once.

### 6. The 2-D launch costs index arithmetic, not geometry

`cuda_2d` is coalesced (4 sectors/request) but executes 5.4× the instructions of `cuda_flat`:
64-bit row × n products and bounds checks in a grid-y loop that nvcc unrolls 4×. It waits
on memory less (long-scoreboard stall 3.2 vs 19.2 warm) and issues more. DRAM-bound FP32
hides this (100.4% of copy at 8192²); FP16 (73.1%) and the L2 regime (22.5 vs 12.4 µs warm)
expose it. A 2-D kernel without the grid-y loop would cost less; the geometry itself is
not the problem.

### 7. Triton matches CUDA when DRAM-bound

Triton vectorizes both layouts to 16-byte accesses on its own (SASS below). DRAM-bound,
`triton` is within 1.3% of `cuda_vec` and `triton_2d` within 2.6% (56 registers in FP32,
63.7% warps active). Its call costs about 4 µs more than the CUDA extension, which shows
at small sizes. Warm FP32 `triton` is slower than `cuda_vec` (15.8 vs 9.5 µs, L2 49.5%
busy); FP16 matches (5.5 µs). These captures do not establish why.

### The FP16 `cuda_flat` anomaly near the boundary

In the benchmark, FP16 `cuda_flat` beats memcpy at 3584² (40.2 vs 49.6 µs) and 4096²
(67.6 vs 79.5 µs), though it is 1.9× slower than `cuda_vec` cold. The result is bimodal:
its run medians were 40.2/40.2/**56.9** µs at 3584² and **83.6**/67.6/67.4 µs at 4096², and
`steady_state.py` measured 40.4 µs (3584²) and 56.7 µs (4096²), again ahead of memcpy
(46.7 and 81.2 µs). With a reused output block it disappears
(`cuda_vec` 26.0 µs vs `cuda_flat` 48.4 µs at 4096²). It is an interaction of
the slower, narrower kernel with which lines L2 retains between calls when the footprint
just exceeds L2. ncu cannot show which: its replay passes save and restore memory, which
leaves read hit rates at 0.1% even in a steady-state capture
(`artifacts/matrix_copy/steady/`). Treat cases within ~1.5× of the L2 size as unstable.

## SASS

Static global loads and stores per kernel ([inspect_sass.py](../benchmarks/inspect_sass.py),
`artifacts/matrix_copy/sass/counts.json`), FP32 instantiation (FP16 is the same with
`.U16` scalars):

| Kernel | Loads | Stores |
|---|---|---|
| `flat_kernel` | 1 `LDG.E.CONSTANT` | 1 `STG.E` |
| `rows_kernel` / `columns_kernel` | 5 `LDG.E.CONSTANT` (loop unrolled 4× + remainder) | 5 `STG.E` |
| `pack_kernel` (`cuda_vec`) | 1 `LDG.E.128.CONSTANT` + 1 scalar tail | 1 `STG.E.128` + 1 tail |
| `ilp_kernel<2>` / `<4>` | 2 / 4 `LDG.E.128.CONSTANT` + tail | 2 / 4 `STG.E.128` + tail |
| `grid_stride_kernel` | 5 `LDG.E.128.CONSTANT` (unrolled 4×) + tail | 5 `STG.E.128` + tail |
| `streaming_kernel` | 1 `LDG.E.EF.128` + 1 `LDG.E` tail | 1 `STG.E.EF.128` + 1 `STG.E` tail |
| Triton flat, 32 / 16-bit | 2 / 1 `LDG.E.128` | 2 / 1 `STG.E.128` |
| Triton 64 × 64 tile, 32 / 16-bit | 8 / 4 `LDG.E.128` | 8 / 4 `STG.E.128` |

`cuda_vec` emits `LDG.E.128` / `STG.E.128` as intended. Static registers
(`resources.txt`) range from 8 (`flat_kernel`) to 40 (`ilp_kernel<4>`). ncu reports 16
for `cuda_flat` at launch (allocation granularity) and the static count elsewhere.

## Answers

- **How close to memcpy?** DRAM-bound FP32: every coalesced kernel reaches 93–100% of
  memcpy (4096²: `cuda_flat` 98.6%). FP16: 16-byte kernels 95–98%, 2-byte scalar kernels
  73–92%. memcpy itself reaches 85–86% of 960 GB/s. In L2,
  memcpy is ahead mostly by its smaller call cost.
- **Where is the L2 cliff?** Where the resident footprint crosses 64 MiB: 3·N²·e under
  `lab.bench` (FP32 between 2304² and 2560², FP16 between 3072² and 3584²) and 2·N²·e
  with a reused output (FP32 between 2896² and 3072², FP16 between 4096² and 4608²).
  The source's 4096² FP32 is DRAM-bound in both.
- **How bad is uncoalesced?** 2.0× slower than memcpy (FP32) and 5.7× (FP16) at DRAM-bound
  sizes, up to 19× when the data is in L2. The cost is L2 transactions; DRAM traffic
  stays near the minimum.
- **Do vectorization and ILP still matter?** Vectorization: not for DRAM-bound FP32, yes
  for FP16 (1.9× kernel time) and for any L2-resident size. ILP and grid-stride loops: no,
  slightly slower everywhere.
- **Recommended kernel:** `cuda_vec`. It is within 3% of memcpy DRAM-bound in every dtype,
  is the fastest kernel in L2, and is the simplest pack variant.
- **For 08 · Transpose:** the copy baselines at 4096² FP32 are 162.2 µs (memcpy), 164.5 µs
  (`cuda_flat`) and 174.7 µs (`triton_2d`, the same 64 × 64 tile).

## Limitations

- One desktop GPU, unlocked clocks; benchmark samples time whole Python calls.
- ncu runs at base clocks with one launch per report, and its replay destroys L2 state at
  sizes near the L2 capacity, so warm captures are limited to 2048².
- `pytorch` (memcpy) has no kernel for ncu to capture.
- The steady-state cliff table is from single runs, not three.
