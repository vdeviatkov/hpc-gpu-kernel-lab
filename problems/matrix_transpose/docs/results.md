# Results

[← Matrix Transpose](../README.md) · [Design](design.md) · **Results** · [Testing](testing.md) ·
[Benchmarking](benchmarking.md) · [Profiling](profiling.md)

**Session:** 2026-11-02 PST (19:23 UTC), cuda.0, RTX 5080 (84 SMs, 64 MiB L2), CUDA toolkit
13.4.92, PyTorch 2.14.1 (CUDA 13.0), Triton 3.8.0, driver 615.71.09, Nsight Compute
2026.3.1, Nsight Systems 2026.3.2. Clocks unlocked; no other compute processes. Every case
passed its correctness gate before timing; the final suite is 500 passed, no skips, and
Compute Sanitizer reports 0 errors and 0 hazards (see [testing](testing.md)).

Tables show the **median of three run medians**, each from 100 CUDA-event samples after 25
warmups. Inputs are reused and output allocation is included. The run-median spread is
0.3% (median); 24 of 260 backend × dtype × size cells exceed 5%, all at 512² to 2560², where
a call is 6–60 µs and the L2 boundary sits. The JSON records no commit: the code ran from a
synced copy, hashed in `artifacts/matrix_transpose/remote_source_sha256.txt` (identical to
the sources here). Compact evidence: [measured_summary.json](../benchmarks/measured_summary.json).

> **Copy ceilings are not competitors.** In every table the rows under *Copy ceilings* —
> `memcpy` and copy `cuda_vec` — are plain contiguous copies of the input (output has the
> input's shape and layout, no transpose) and are not checked against the transposed
> reference. They mark the bandwidth ceiling: a transpose moves the same bytes as a copy of
> its input, so it can at best match one. Only the rows under *Transposes* compete.
> - `memcpy` is `a.clone()` in the harness, a driver `cudaMemcpyAsync` device-to-device with
>   PyTorch's small call cost.
> - copy `cuda_vec` is 07 · Matrix Copy's 16-byte pack kernel, called through the same
>   extension path as the CUDA transposes, so it is the like-for-like ceiling where call
>   overhead matters (in L2).

![Effective bandwidth against logical traffic per call. Past the L2 boundary the tiled CUDA kernels and Triton sit at about 740–750 GB/s (FP32) and 705–727 GB/s (FP16), just under the plain-copy ceiling (memcpy, 815–820; not a transpose); the unpadded tile trails in FP16, pytorch stays near 360/280 and naive_row near 510 (FP32) and 140 GB/s (FP16). Below the boundary cuda_vec_ilp tracks the copy cuda_vec ceiling, Triton sits between it and the 32×32 tiles, and the unpadded tile falls behind.](../figures/bandwidth_vs_size.svg)

## DRAM-bound: how close to the copy ceiling

Source case, 7000 × 6000. Latency per call (µs), effective GB/s, and the ceiling's latency
over the backend's (100% = as fast).

| Backend | FP32 µs | GB/s | vs memcpy | vs copy `cuda_vec` | FP16 µs | GB/s | vs memcpy | vs copy `cuda_vec` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ***Copy ceilings (plain copies, not transposes)*** | | | | | | | | |
| memcpy | 412.1 | 815 | 100% | 100% | 205.3 | 818 | 100% | 101% |
| copy `cuda_vec` | 413.3 | 813 | 100% | 100% | 208.4 | 806 | 99% | 100% |
| ***Transposes*** | | | | | | | | |
| `pytorch` | 943.0 | 356 | 44% | 44% | 597.6 | 281 | 34% | 35% |
| `cuda_naive_row` | 659.9 | 509 | 62% | 63% | 1226.2 | 137 | 17% | 17% |
| `cuda_naive_col` | 977.5 | 344 | 42% | 42% | 517.6 | 325 | 40% | 40% |
| `cuda_smem` | 454.2 | 740 | 91% | 91% | 247.4 | 679 | 83% | 84% |
| `cuda_smem_pad` | 453.2 | 741 | 91% | 91% | 237.9 | 706 | 86% | 88% |
| `cuda_smem_swizzle` | 453.2 | 741 | 91% | 91% | 237.2 | 708 | 87% | 88% |
| `cuda_vec_ilp` | 453.0 | 742 | 91% | 91% | 233.1 | 721 | 88% | 89% |
| `triton` | 450.1 | 747 | 92% | 92% | 231.1 | 727 | 89% | 90% |

- **The best transposes reach 91–92% of the copy ceiling in FP32 and 88–89% in FP16.** The
  four tiled CUDA kernels are within 0.3% of each other in FP32; Triton is 0.6% ahead.
  Kernel-only times from the Nsight Systems timeline (10 launches each, same clocks) agree:
  copy `cuda_vec` 408.7 µs against 437.7 (Triton) and 440.1–444.6 µs (CUDA tiles) in FP32,
  92–93%; 207.6 against 216.5 (Triton, 96%) and 222.1 (`cuda_vec_ilp`, 93%) in FP16.
- **Every tiled kernel is about 2× PyTorch.** `a.t().clone()` runs a generic strided
  `elementwise_kernel` (895.6 µs FP32, 588.8 µs FP16 in the timeline): 2.1× slower than the
  tiles per call in FP32, 2.4–2.6× in FP16.
- **The tiles move the minimum DRAM traffic.** Cold ncu captures read 168.0 MB in FP32 and
  84.0–84.3 MB in FP16 for the 32×32 tiles and Triton, exactly the input; writes are
  136–141 MB (FP32) because dirty lines still sit in L2 when the kernel ends, as for the
  copy (128.5 MB). A single cold capture occasionally reads up to 8.7% more and runs about
  8% longer, for any kernel ([profiling](profiling.md#reading-the-evidence));
  `cuda_smem_swizzle` FP32 did so in this sweep (182.6 MB, 422.7 µs; three re-captures:
  168.3/182.6/168.0 MB, 393/423/393 µs).
- At 8192² FP32 the tiles reach 785–787 GB/s and Triton 799, 96–98% of memcpy's 819. Why a
  transpose trails a plain copy by 6–12% when both move the minimum bytes is not yet
  explained (see [open questions](#open-questions)).

## Bank conflicts: the primary lesson

Counters from single cold launches at 7000 × 6000 (ncu, base clocks, caches flushed).
`ld wavefronts/inst` is shared-load wavefronts per shared-load instruction: 1.00 is
conflict-free for 4-byte accesses, 32 means every access in a warp was serialized.

| Kernel | FP32 µs | smem ld conflicts | ld wavefronts/inst | FP16 µs | smem ld conflicts | ld wavefronts/inst |
|---|---:|---:|---:|---:|---:|---:|
| `cuda_smem` | 405.2 | 40,891,468 | 32.12 | 231.5 | 19,866,562 | 16.12 |
| `cuda_smem_pad` | 391.7 | 597 | 1.00 | 218.0 | 662,980 | 1.50 |
| `cuda_smem_swizzle` | 422.7* | 533 | 1.00 | 216.9 | 1,342 | 1.00 |

\* A slow single capture; three re-captures took 393, 423 and 393 µs.

- **The hypothesis holds exactly**, in both measurement sessions. Unpadded column reads are
  32-way in FP32 and 16-way in FP16, as the [bank model](design.md#3-simple-performance-model)
  predicted. The per-SASS counters of the full reports agree: the one `LDS` of `cuda_smem`
  FP32 issues 42.0 M wavefronts against an ideal of 1.31 M (32.0×). Shared stores are
  conflict-free in all three.
- **Padding fixes FP32 but leaves FP16 at 1.5 wavefronts per load**, as modelled: a 2-byte
  pad element is half a bank word, so some columns still pair up (worst case 2-way,
  average 1.5). The swizzle, which XORs whole 4-byte words, is conflict-free in both.
- **When DRAM-bound, the conflicts barely cost time.** At the source shape `cuda_smem` is
  within 0.2% of the padded tile in FP32 (454.2 vs 453.2 µs per call) and 4% slower in
  FP16 (247.4 vs 237.9). The kernel waits on DRAM, and 32 serialized shared wavefronts per
  load hide behind it. Under ncu's base clocks the gap is 3.4% (FP32) and 6.2% (FP16).
- **When L2-resident, they cost up to 1.8×.** At 2048² FP32 (warm ncu) `cuda_smem` takes
  27.9 µs against 15.2 for the padded tile; per call, 1031 against 1504 GB/s.
- **Padding and swizzling are indistinguishable in time,** even in FP16 where the padded
  tile still has 1.5 wavefronts per load (237.9 vs 237.2 µs per call; 12.3 vs 12.4 µs
  warm). The extra half wavefront is hidden.

## Which side of a naive transpose hurts more?

| Kernel | Coalesced | FP32 7000×6000 µs | FP16 7000×6000 µs | FP32 2048² warm µs | FP16 2048² warm µs |
|---|---|---:|---:|---:|---:|
| `cuda_naive_row` | loads (stores 31.9 sectors/request) | 659.9 | 1226.2 | 58.9 | 113.4 |
| `cuda_naive_col` | stores (loads 32.0 sectors/request) | 977.5 | 517.6 | 29.2 | 24.4 |

(First two columns: benchmark per call; last two: ncu kernel time.)

- **It depends on dtype and regime; strided stores are the worse problem overall.** Only
  FP32 from DRAM favours `cuda_naive_row` (strided stores, 1.5× faster). In FP16, and in
  both dtypes when L2-resident, strided stores are 2–5× slower than strided loads.
- Strided loads are partly rescued by caches: a block's 8 warps read 8 neighbouring
  columns, so each 32-byte sector fetched for one warp is reused by the others (L2 hit
  52% FP16; `cuda_naive_col` reaches 1230 GB/s at 2560² FP32). Every strided store is a
  separate partial-sector write to L2.
- `cuda_naive_row` in FP16 is the slowest kernel measured (137 GB/s, 17% of the memcpy
  ceiling), 1.9× slower than in FP32 for the same number of store requests. The likely
  cause is the cost of 2-byte partial writes in L2, which is not yet tested (see open
  questions).

## Below the L2 boundary

Effective GB/s per call (every size is in [measured_summary.json](../benchmarks/measured_summary.json)).
memcpy's lead over copy `cuda_vec` here is its ~4 µs smaller fixed call cost (6.3 vs 10.4 µs
at 512² FP32), so copy `cuda_vec` is the fair ceiling in this range.

| FP32 | 1024² | 2048² | 2304² | 2560² | 4096² | 8192² |
|---|---:|---:|---:|---:|---:|---:|
| ***Copy ceilings (plain copies, not transposes)*** | | | | | | |
| memcpy | 1365 | 2759 | 1900 | 905 | 827 | 819 |
| copy `cuda_vec` | 794 | 1967 | 2077 | 922 | 798 | 814 |
| ***Transposes*** | | | | | | |
| `cuda_smem` | 514 | 1031 | 1152 | 1174 | 782 | 785 |
| `cuda_smem_pad` | 648 | 1504 | 1679 | 950 | 784 | 787 |
| `cuda_smem_swizzle` | 566 | 1507 | 1681 | 957 | 786 | 786 |
| `cuda_vec_ilp` | 686 | 1830 | 2011 | 922 | 783 | 766 |
| `triton` | 558 | 1496 | 1676 | 889 | 773 | 799 |

| FP16 | 1024² | 2048² | 3072² | 3584² | 4096² | 8192² |
|---|---:|---:|---:|---:|---:|---:|
| ***Copy ceilings (plain copies, not transposes)*** | | | | | | |
| memcpy | 672 | 2081 | 2703 | 1103 | 839 | 823 |
| copy `cuda_vec` | 401 | 1216 | 2055 | 1012 | 803 | 812 |
| ***Transposes*** | | | | | | |
| `cuda_smem` | 295 | 718 | 974 | 1031 | 867 | 671 |
| `cuda_smem_pad` | 325 | 850 | 1229 | 973 | 776 | 717 |
| `cuda_smem_swizzle` | 328 | 864 | 1198 | 1012 | 767 | 717 |
| `cuda_vec_ilp` | 383 | 1172 | 1859 | 925 | 785 | 731 |
| `triton` | 277 | 976 | 1636 | 851 | 725 | 739 |

- **With data in L2, `cuda_vec_ilp` is the fastest transpose per call,** at 90–97% of the
  copy `cuda_vec` ceiling (2011 vs 2077 GB/s at 2304² FP32; 1859 vs 2055 at 3072² FP16) and
  1.2–1.5× the 32×32 tiles. Its 16-byte global accesses need a quarter (FP32) or an eighth
  (FP16) of the global load/store instructions, which matters once DRAM no longer limits.
- **The L2 cliff falls where 3·M·N·e crosses 64 MiB,** as for the copy: between 2304² and
  2560² in FP32, 3072² and 3584² in FP16. Kernels slower than the cliff bandwidth
  (`cuda_smem`, the padded tiles in FP16) appear to "peak" past it.
- The 1024² `cuda_smem_swizzle` FP32 value (566) has a 15% run spread; the padded tile's
  648 is the better estimate of the 32×32 tile there.

## Triton

`triton` loads a 64 × 64 tile indexed [input column, input row], which is already the output
order, and lets the compiler move it between register layouts
([implementation](../triton/implementation.py)). No swizzle or padding is written by hand.

- **DRAM-bound, it is the fastest transpose,** narrowly: 450.1 µs FP32 and 231.1 µs FP16
  per call, 0.6–0.9% ahead of `cuda_vec_ilp`. Kernel-only, the margin is clearer in FP16:
  216.5 vs 222.1 µs in the timeline, and 172.3 vs 194.2 µs in single ncu captures, where
  `cuda_vec_ilp` read 91.7 MB against Triton's 84.3.
- **In L2, it sits between the 32×32 tiles and `cuda_vec_ilp` per call:** level with the
  padded tile in FP32 (1496 vs 1504 GB/s at 2048²), 1.3× it in FP16 (1636 vs 1229 at
  3072²), and 82–88% of `cuda_vec_ilp`. Kernel-only it is level with `cuda_vec_ilp`: warm
  ncu at 2048² gives 11.8 vs 10.6 µs (FP32) and 6.1 vs 6.9 µs (FP16). The per-call gap is
  its larger call cost.
- **Its call costs about 4 µs more than the CUDA extension:** 15.0–15.1 µs at 512² and
  1024², against 10.7–11.2 µs (CUDA) and 6.1–6.3 µs (memcpy), as for 07's Triton copy.
- **Conflict-free by construction.** The compiler routes the layout change through 8 KB
  (FP32) or 4 KB (FP16) of dynamic shared memory in two half-tile passes, writing with
  `STSM` (store-matrix) and reading with `LDS.128`. Every one of these instructions issues
  exactly its ideal wavefront count in the full reports. Like `cuda_vec_ilp`, the aggregate
  counters still report some conflicts (next section); the aggregate store ratio reads
  0.00 because `STSM` is not counted as a shared-store instruction, and 4.7 load
  wavefronts per instruction is the ideal 4 of a 128-bit `LDS` plus that aggregate excess.
- It uses 72 (FP32) and 62 (FP16) registers per thread at 128 threads per program, so only
  56–63% of warp slots are active when DRAM-bound; it still keeps DRAM as busy as the CUDA
  tiles.

## Shared-memory counters of `cuda_vec_ilp` and Triton

The aggregate counters report a few conflicts for both kernels (`cuda_vec_ilp` FP32 cold:
36,918 load and 26,748 store conflicts, 1.02–1.03 wavefronts per instruction; 1.05–1.39
warm; Triton FP32 cold: 235,837 load conflicts), while the bank model predicts none. The
per-instruction source counters of the full reports show **zero excess wavefronts** on
every pack-path `STS`/`LDS` of `cuda_vec_ilp` and every `STSM`/`LDS.128` of Triton, in both
dtypes. The aggregate excess therefore does not come from the shared addresses. It is
unexplained; one candidate is arbitration with the kernels' 128-bit global traffic, which
shares L1's data banks.

## SASS

From `cuobjdump` of the extension and the Triton full reports:

| Kernel | Global loads | Global stores | Shared | Regs (FP32/FP16) | Shared B/block (FP32) |
|---|---|---|---|---:|---:|
| `cuda_naive_*` | `LDG.E` / `.U16`, 5 unrolled | `STG.E` / `.U16`, 5 unrolled | — | 34 / 34 | 0 |
| `cuda_smem*` | `LDG.E` / `.U16` | `STG.E` / `.U16` | 1 `STS`, 1 `LDS`, 2 `BAR` | 23 / 23 | 4096 (4224 padded) |
| `cuda_vec_ilp` | `LDG.E.128` × 4 (FP16 × 2) | `STG.E.128` × 4 (FP16 × 2) | 16 scalar `STS` + 16 `LDS`, 1 `BAR` | 44 / 36 | 16384 |
| `triton` | `LDG.E.128` × 8 (FP16 × 4) | scalar `STG.E` / `.U16` × 32 | `STSM.16.M88.4` + `LDS.128`, 3 `BAR` | 72 / 62 | 8192 dynamic |

The two fastest transposes vectorize opposite sides. `cuda_vec_ilp` moves 16 bytes per
global access on both sides, but its swizzle scatters elements, so shared memory is
accessed one element at a time. Triton vectorizes shared memory with matrix stores and
128-bit loads, but writes global memory one element at a time (still coalesced: 4.0 and
2.5 sectors per request). Both reach the same DRAM-bound speed, and their kernel times in
L2 are level.

## Answers

- **Are global loads and stores both coalesced?** In every tiled kernel, yes: 4.0 (FP32) or
  2.0–2.5 (FP16) sectors per scalar request, 16 per 128-bit request. The naive kernels
  coalesce one side and touch 32 sectors per request on the other.
- **Does padding change measured conflicts and latency?** Conflicts, exactly as modelled
  (32.12 → 1.00 wavefronts per load in FP32, 16.12 → 1.50 in FP16). Latency only off DRAM:
  1.8× at 2048² FP32 in L2, 0.2% (FP32) and 4% (FP16) at the source shape.
- **How close is performance to the copy control?** 91–92% of the copy ceiling in FP32 and
  88–89% in FP16 at 7000 × 6000; 96–98% at 8192² FP32; 90–97% of copy `cuda_vec` in L2 for
  `cuda_vec_ilp`.

## Limitations

- One GPU, one session, unlocked clocks. ncu durations are single launches at base clocks
  and vary by up to 8% between captures; compare them within a capture set, not across.
- Every sample includes Python dispatch, argument checks and output allocation (about
  10 µs, 15 µs for Triton). Small sizes measure the call, not the kernel.
- BF16 runs the same kernels as FP16 and is tested but not timed.
- Triton's tile (64 × 64) and warps (4) are fixed defaults, not tuned; the CUDA tiles are
  not tuned either.

## Open questions

1. **Why does a minimum-traffic transpose trail copy by 6–12%?** Both read and write the
   same DRAM bytes, but a tile's output goes to 32 (or 256) different rows, so DRAM page
   locality of the writes is worse. Test: tiles with longer output-row segments (e.g.
   32×128), comparing per-call latency and DRAM read/write throughput against the copy.
2. **Why is a 2-byte strided store 1.9× slower than a 4-byte one?** Test: L2 write
   transactions and throughput (`lts__t_sectors_op_write`, `lts__throughput`) for
   `cuda_naive_row` FP32 vs FP16 at the same element count.
3. **What are the swizzled kernels' aggregate conflicts?** Test: force `cuda_vec_ilp`'s
   scalar global path on an aligned shape (unchanged shared pattern, no 128-bit global
   loads); if the aggregate conflicts vanish, they come from global traffic, not from
   shared addresses.
4. **Would Triton gain from 16-byte stores or a larger tile?** Test: sweep `block_m`,
   `block_n` and `num_warps`, and check in SASS whether the stores become `STG.E.128`.
