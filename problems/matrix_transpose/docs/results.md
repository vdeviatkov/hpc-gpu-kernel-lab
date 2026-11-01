# Results

[← Matrix Transpose](../README.md) · [Design](design.md) · **Results** ·
[Benchmarking](benchmarking.md) · [Profiling](profiling.md)

**Session:** 2026-10-31 PDT (2026-11-01 01:52 UTC), cuda.0, RTX 5080 (84 SMs, 64 MiB L2),
CUDA toolkit 13.4.92, PyTorch 2.14.1 (CUDA 13.0), driver 615.71.09, Nsight Compute
2026.3.1, Nsight Systems 2026.3.2. Clocks unlocked; no other compute processes. Every case
passed its correctness gate before timing: 399 tests passed (42 Triton cases skip: stub),
and Compute Sanitizer memcheck, racecheck, synccheck and initcheck report 0 errors and 0
hazards over the 349 CUDA tests.

Tables show the **median of three run medians**, each from 100 CUDA-event samples after 25
warmups. Inputs are reused and output allocation is included. The run-median spread is
0.2% (median); three small L2-resident cases reach 5–10%. The JSON records no commit: the
code ran from a synced copy, hashed in `artifacts/matrix_transpose/remote_source_sha256.txt`
(identical to the sources here). Compact evidence:
[measured_summary.json](../benchmarks/measured_summary.json).

![Effective bandwidth against logical traffic per call. Past the L2 boundary the three padded/swizzled/vector tiled kernels sit at about 745 GB/s (FP32) and 710 GB/s (FP16), just under the plain-copy ceiling (memcpy, 820; not a transpose); the unpadded tile trails in FP16, pytorch stays near 360/280 and naive_row near 510 (FP32) and 140 GB/s (FP16). Below the boundary cuda_vec_ilp tracks the copy cuda_vec ceiling and the unpadded tile falls behind.](../figures/bandwidth_vs_size.svg)

## Source case, 7000 × 6000: how close is transpose to copy?

Latency per call (µs), effective GB/s, and the reference's latency over the backend's
(100% = as fast).

> **Copy ceilings are not competitors.** In every table here the rows under *Copy ceilings*
> — `memcpy` and copy `cuda_vec` — are plain contiguous copies of the input (output has the
> input's shape and layout, no transpose) and are not checked against the transposed
> reference. They mark the bandwidth ceiling: a transpose moves the same bytes as a copy of
> its input, so it can at best match one. Only the rows under *Transposes* compete.
> - `memcpy` is `a.clone()` in the harness, a driver `cudaMemcpyAsync` device-to-device with
>   PyTorch's small call cost.
> - copy `cuda_vec` is 07 · Matrix Copy's 16-byte pack kernel, called through the same
>   extension path as the transposes, so it is the like-for-like ceiling where call overhead
>   matters (in L2).

| Backend | FP32 µs | GB/s | vs memcpy | vs copy `cuda_vec` | FP16 µs | GB/s | vs memcpy | vs copy `cuda_vec` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ***Copy ceilings (plain copies, not transposes)*** | | | | | | | | |
| memcpy | 409.5 | 821 | 100% | 100% | 205.0 | 819 | 100% | 101% |
| copy `cuda_vec` | 411.3 | 817 | 100% | 100% | 207.5 | 810 | 99% | 100% |
| ***Transposes*** | | | | | | | | |
| `pytorch` | 941.3 | 357 | 44% | 44% | 596.6 | 282 | 34% | 35% |
| `cuda_naive_row` | 656.6 | 512 | 62% | 63% | 1221.1 | 138 | 17% | 17% |
| `cuda_naive_col` | 976.4 | 344 | 42% | 42% | 516.5 | 325 | 40% | 40% |
| `cuda_smem` | 452.0 | 743 | 91% | 91% | 247.3 | 679 | 83% | 84% |
| `cuda_smem_pad` | 451.2 | 745 | 91% | 91% | 237.1 | 709 | 86% | 87% |
| `cuda_smem_swizzle` | 451.1 | 745 | 91% | 91% | 237.1 | 709 | 86% | 88% |
| `cuda_vec_ilp` | 451.9 | 743 | 91% | 91% | 233.5 | 719 | 88% | 89% |

- **The best transpose reaches 91% of the memcpy ceiling in FP32 and 88% in FP16** (`cuda_vec_ilp`,
  233.5 vs 205.0 µs). Kernel-only times from the Nsight Systems timeline (10 launches, same
  clocks) give the same picture: the copy `cuda_vec` ceiling 408.0 µs against 433–442 µs for the tiled
  FP32 kernels (92–94%), and 203.8 against 220.5 µs for FP16 `cuda_vec_ilp` (92%).
- **Every tiled kernel is about 2× PyTorch.** `a.t().clone()` runs a generic strided
  `elementwise_kernel` (896 µs FP32 in the timeline): 2.1× slower than the tiles in FP32,
  2.4–2.6× in FP16.
- **The tiles move the minimum DRAM traffic.** In cold ncu captures every tiled kernel
  reads 168.0 MB (FP32), exactly the input; writes are 139.7–141.1 MB because dirty lines
  still sit in L2 when the kernel ends, as for the copy (129.3 MB).
- At 8192² FP32 the tiles reach 787 GB/s, 96% of the memcpy ceiling's 821. Why transpose trails a plain copy by 6–12% when both move the minimum bytes is not
  yet explained (see [open questions](#open-questions)).

## Bank conflicts: the primary lesson

Counters from single cold launches at 7000 × 6000 (ncu, base clocks, caches flushed).
`ld wavefronts/inst` is shared-load wavefronts per shared-load instruction: 1.00 is
conflict-free, 32 means every access in a warp was serialized.

| Kernel | FP32 µs | smem ld conflicts | ld wavefronts/inst | FP16 µs | smem ld conflicts | ld wavefronts/inst |
|---|---:|---:|---:|---:|---:|---:|
| `cuda_smem` | 406.5 | 40,885,604 | 32.12 | 230.5 | 19,869,038 | 16.12 |
| `cuda_smem_pad` | 392.5 | 539 | 1.00 | 215.8 | 662,964 | 1.50 |
| `cuda_smem_swizzle` | 392.6 | 548 | 1.00 | 216.3 | 1,286 | 1.00 |

- **The hypothesis holds exactly.** Unpadded column reads are 32-way in FP32 and 16-way in
  FP16, as the [bank model](design.md#3-simple-performance-model) predicted. The per-SASS
  counters of the full reports agree: the one `LDS` of `cuda_smem` FP32 issues 42.0 M
  wavefronts against an ideal of 1.31 M (32.0×). Shared stores are conflict-free in all
  three.
- **Padding fixes FP32 but leaves FP16 at 1.5 wavefronts per load**, as modelled: a 2-byte
  pad element is half a bank word, so some columns still pair up (worst case 2-way,
  average 1.5). The swizzle, which XORs whole 4-byte words, is conflict-free in both.
- **When DRAM-bound, the conflicts barely cost time.** At the source shape `cuda_smem` is
  within 0.2% of the padded tile in FP32 (452.0 vs 451.2 µs per call) and 4% slower in
  FP16. The kernel waits on DRAM, and 32 serialized shared wavefronts per load hide behind
  it. Under ncu's base clocks the gap is 3.5% (FP32) and 7% (FP16).
- **When L2-resident, they cost up to 1.8×.** At 2048² FP32 (warm ncu) `cuda_smem` takes
  27.2 µs against 15.0 for the padded tile; per call, 1035 against 1507 GB/s.
- **Padding and swizzling are indistinguishable in time,** even in FP16 where the padded
  tile still has 1.5 wavefronts per load (237.1 µs both; 12.4 µs both warm). The extra
  half wavefront is hidden.

## Which side of a naive transpose hurts more?

| Kernel | Coalesced | FP32 7000×6000 µs | FP16 7000×6000 µs | FP32 2048² warm µs | FP16 2048² warm µs |
|---|---|---:|---:|---:|---:|
| `cuda_naive_row` | loads (stores 31.9 sectors/request) | 656.6 | 1221.1 | 58.7 | 113.2 |
| `cuda_naive_col` | stores (loads 32.0 sectors/request) | 976.4 | 516.5 | 28.3 | 24.4 |

(First two columns: benchmark per call; last two: ncu kernel time.)

- **It depends on dtype and regime; strided stores are the worse problem overall.** Only
  FP32 from DRAM favours `cuda_naive_row` (strided stores, 1.5× faster). In FP16, and in
  both dtypes when L2-resident, strided stores are 2–5× slower than strided loads.
- Strided loads are partly rescued by caches: a block's 8 warps read 8 neighbouring
  columns, so each 32-byte sector fetched for one warp is reused by the others (L2 hit
  55% FP16; `cuda_naive_col` reaches 1230 GB/s at 2560² FP32). Every strided store is a
  separate partial-sector write to L2.
- `cuda_naive_row` in FP16 is the slowest kernel measured (138 GB/s, 17% of the memcpy ceiling), 1.9×
  slower than in FP32 for the same number of store requests. The likely cause is the cost
  of 2-byte partial writes in L2, which is not yet tested (see open questions).

## Below the L2 boundary

Effective GB/s per call (the full sweep is in [measured_summary.json](../benchmarks/measured_summary.json)).
Rows under *Copy ceilings* are plain copies, not transposes; memcpy's lead over copy
`cuda_vec` here is its ~4 µs smaller fixed call cost (6.0 vs 10.2 µs at 512² FP32), so copy
`cuda_vec` is the fair ceiling in this range.

| FP32 | 1024² | 2048² | 2304² | 2560² | 4096² | 8192² |
|---|---:|---:|---:|---:|---:|---:|
| ***Copy ceilings (plain copies, not transposes)*** | | | | | | |
| memcpy | 1365 | 2752 | 2269 | 992 | 832 | 821 |
| copy `cuda_vec` | 809 | 2044 | 2112 | 955 | 802 | 817 |
| ***Transposes*** | | | | | | |
| `cuda_smem` | 515 | 1035 | 1153 | 1176 | 778 | 789 |
| `cuda_smem_pad` | 665 | 1507 | 1687 | 955 | 787 | 787 |
| `cuda_smem_swizzle` | 681 | 1504 | 1728 | 904 | 788 | 787 |
| `cuda_vec_ilp` | 686 | 1836 | 2082 | 955 | 787 | 764 |

| FP16 | 1024² | 2048² | 3072² | 3584² | 4096² | 8192² |
|---|---:|---:|---:|---:|---:|---:|
| ***Copy ceilings (plain copies, not transposes)*** | | | | | | |
| memcpy | 660 | 2072 | 2853 | 1012 | 845 | 825 |
| copy `cuda_vec` | 406 | 1364 | 2055 | 930 | 805 | 815 |
| ***Transposes*** | | | | | | |
| `cuda_smem` | 294 | 744 | 975 | 1032 | 867 | 680 |
| `cuda_smem_pad` | 342 | 823 | 1211 | 1300 | 776 | 723 |
| `cuda_smem_swizzle` | 341 | 824 | 1204 | 1302 | 768 | 725 |
| `cuda_vec_ilp` | 390 | 1173 | 1864 | 938 | 793 | 742 |

- **With data in L2, `cuda_vec_ilp` is the fastest transpose,** within 1–10% of the copy
  `cuda_vec` ceiling (2082 vs 2112 GB/s at 2304² FP32; 1864 vs 2055 at 3072² FP16) and 1.2–1.5×
  the 32×32 tiles. Its 16-byte global accesses need a quarter (FP32) or an eighth (FP16)
  of the load/store instructions, which matters once DRAM no longer limits.
- **The L2 cliff falls where 3·M·N·e crosses 64 MiB,** as for the copy: between 2304² and
  2560² in FP32, 3072² and 3584² in FP16. Kernels slower than the cliff bandwidth
  (`cuda_smem`, the padded tiles in FP16) appear to "peak" past it.

## Shared-memory counters of `cuda_vec_ilp`

The aggregate counters report a few conflicts for `cuda_vec_ilp` (FP32 cold: 36,180 load
and 27,005 store conflicts, 1.02–1.03 wavefronts per instruction; 1.05–1.38 warm), while
the bank model predicts none. The per-instruction source counters of the full report show
**zero excess wavefronts** on all 32 pack-path `STS`/`LDS` instructions in both dtypes:
each issues exactly its ideal count. The aggregate excess therefore does not come from
the swizzled addresses. It is unexplained; one candidate is arbitration with the
kernel's 128-bit global traffic, which shares L1's data banks.

## Open questions

1. **Why does a minimum-traffic transpose trail copy by 6–12%?** Both read and write the
   same DRAM bytes, but a tile's output goes to 32 (or 256) different rows, so DRAM page
   locality of the writes is worse. Test: tiles with longer output-row segments (e.g.
   32×128), comparing per-call latency and DRAM read/write throughput against the copy.
2. **Why is a 2-byte strided store 1.9× slower than a 4-byte one?** Test: L2 write
   transactions and throughput (`lts__t_sectors_op_write`, `lts__throughput`) for
   `cuda_naive_row` FP32 vs FP16 at the same element count.
3. **What are `cuda_vec_ilp`'s aggregate conflicts?** Test: force its scalar global path
   on an aligned shape (unchanged shared pattern, no 128-bit global loads); if the
   aggregate conflicts vanish, they come from global traffic, not from shared addresses.
