# Profiling

[← RGB to Grayscale](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · **Profiling**

Lab-wide workflow and `lab.profile`: [profiling](../../../docs/profiling.md). The same
analysis order is explained in more depth for
[Reverse Array](../../reverse_array/docs/profiling.md).

## Capturing one launch

```bash
# DRAM-bound: 4096 x 4096, caches flushed before the launch (ncu's default)
ncu --profile-from-start off --set full --section SourceCounters --import-source yes \
  -o artifacts/rgb_to_grayscale/ncu/vec_fp16_4096 \
  python -m lab.profile rgb_to_grayscale cuda_vec fp16 h=4096 w=4096 dist=int

# Cache-resident: 2048 x 2048 with caches kept, so the image is in L2 after the warmup
ncu --profile-from-start off --set full --section SourceCounters --import-source yes \
  --cache-control none -o artifacts/rgb_to_grayscale/ncu/vec_fp16_2048_warm \
  python -m lab.profile rgb_to_grayscale cuda_vec fp16 h=2048 w=2048 dist=int
```

`lab.profile` checks the backend against the reference, warms it up (which also compiles
`pytorch_compile`) and runs it once in a capture range, so neither input generation nor
compilation appears. `cuda_planar` produces two kernels per report, the eager reference
five (FP32). [sweep.sh](../benchmarks/sweep.sh) captures all 23 reports used in
[results](results.md): every backend at 4096² in FP32 and FP16 (the reference in FP32
only), plus the CUDA kernels and `pytorch_compile` at 2048² with a warm cache.

ncu locks clocks to base and, by default, flushes caches before the launch, and it times
the kernel alone. Its durations therefore differ from the benchmark's, which time one
Python call. Compare kernels with each other within ncu.

## What to check, in order

A memory-bound kernel can be slow for four reasons. Each row names where to look, the exact
metric, what it means, what to expect here, and what to conclude.

### 1. Does it move more bytes than necessary?

| Metric | Where | Definition | Expected here |
|---|---|---|---|
| `dram__bytes_op_read.sum` | Memory Workload Analysis → DRAM row; Raw page | Bytes read from DRAM by the kernel | `3·P·s` (201 MB at 4096² FP32) |
| `dram__bytes_op_write.sum` | same | Bytes written to DRAM *during* the kernel | ≤ `P·s`; the rest is still in L2 when the kernel ends |
| `lts__t_sectors_srcunit_tex_op_read.sum` | Raw page (L2 "from L1" read sectors) | 32-byte sectors L1 requested from L2 for loads | `3·P·s / 32` (6.29 M at 4096² FP32) |
| `lts__t_sector_hit_rate.pct` | Memory Workload Analysis → L2 | Fraction of L2 sector accesses that hit | ~0% cold at 4096², ~100% warm FP16 2048² |

**Why:** if DRAM reads exceed `3·P·s`, the kernel fetches sectors it does not use, or
fetches the same sector twice, and no amount of tuning elsewhere recovers that. **Decision:**
when DRAM bytes and L2 sectors equal the lower bound, the access pattern wastes nothing at
the DRAM and L2 levels; look at bandwidth use (step 2). Here every CUDA kernel and
`pytorch_compile` meets the lower bound; the eager reference reads 4.3× more.

### 2. Is the stride absorbed before L2?

| Metric | Where | Definition | Expected here |
|---|---|---|---|
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | Memory Workload Analysis → L1/TEX table, Global Load row, *Requests* | Warp-level load instructions that reached L1 | one per load instruction per warp |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | same table, *Sectors* | Sectors those requests touched | — |
| Sectors / request (the ratio) | same table, *Sectors/Req* | How spread out one warp's load is: 4 = 128 B contiguous 4-byte loads | pixel FP32 12, vec 32, smem 16 |
| `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` | same table, *Hit Rate* | Fraction of those sectors found in L1 | pixel 66.7%, vec 50%, smem 0% |
| Avg. Predicated-On Threads Executed | Source page, per instruction | Lanes active for that instruction | 32 for every memory instruction |

**Why:** sectors per request alone looks alarming for `cuda_pixel` (12 instead of 4, "3×
uncoalesced"), but each sector a channel load touches is reused by the next two channel
loads of the same warp. The L1 hit rate shows that reuse: the first load misses, the other
two hit (66.7%). **Decision:** high sectors/request is harmless when the L1 hit rate makes
up the difference and L2 sectors (step 1) stay at the lower bound. It costs L1 throughput
and instruction issue, which only matters once DRAM stops being the limit (step 4).

### 3. Is DRAM kept busy?

| Metric | Where | Definition | Expected here |
|---|---|---|---|
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | GPU Speed Of Light → *DRAM Throughput* | DRAM bandwidth used over the kernel's duration, % of peak | 93–96% for a good kernel at 4096² |
| `gpu__time_duration.sum` | Speed Of Light → *Duration* | Kernel time (ncu clocks) | ~262–265 µs FP32, ~114–133 µs FP16 at 4096² |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | Warp State Statistics → *Stall Long Scoreboard* | Cycles warps wait on a global-memory result, per issued instruction | the top stall |

**Why:** a kernel with the minimum bytes (step 1) can still be slow if it does not keep
enough bytes in flight. **Decision:** at 93–96% DRAM throughput the kernel is at the
hardware limit; differences between such kernels are noise or cache-state effects, not
design. Below ~85%, look for too few bytes in flight (small accesses, low occupancy) or a
saturated unit (step 4).

### 4. Is something else the limit?

| Metric | Where | Definition | Expected here |
|---|---|---|---|
| `smsp__inst_executed.sum` ÷ pixels | Instruction Statistics → *Executed Instructions* | Warp instructions per pixel | vec 0.32, compile 0.80–0.91, pixel 1.06–1.19, smem 2.07–2.54, matmul ~19 |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | Compute Workload Analysis → *Issue Slots Busy* | Cycles in which a warp scheduler issued an instruction | low (2–20%) when DRAM-bound |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | Speed Of Light → *L2 Cache Throughput* | Busiest L2 sub-unit, % of peak | ~31% at 4096²; ~91% for `cuda_vec` warm FP16 2048² |
| `l1tex__throughput.avg.pct_of_peak_sustained_active` | Speed Of Light → *L1/TEX Cache Throughput* | Busiest L1 sub-unit, % of peak | — |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | Warp State Statistics → *Stall Barrier* | Cycles waiting at `__syncthreads()` per issued instruction | only `cuda_smem` (5–23) |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Memory Workload Analysis → Shared Memory table, *Bank Conflicts* | Extra shared-memory cycles caused by conflicts on loads | ~0 FP32; ~1 per `LDS` in FP16 (2-way) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Occupancy → *Achieved Occupancy* | Resident warps, % of the SM maximum | 66–93% |
| `launch__registers_per_thread` | Launch Statistics | Registers per thread | 16 (pixel) to 40 (vec FP32) |

**Why:** when DRAM is not the limit (the warm 2048² FP16 image is entirely in L2) the next
bottleneck appears: L2 bandwidth for a kernel that issues few, wide requests; instruction
issue and L1 for one that issues many narrow ones; barriers and bank conflicts for shared
memory. **Decision:** at a cache-resident size, prefer the kernel that reaches the highest L2
throughput with the fewest instructions per pixel; at a DRAM-bound size, none of these
matter as long as step 3 shows 93%+.

## Reading the planar and reference reports

- `cuda_planar`: compare the second kernel (`rgb_to_grayscale_planar`) with the
  interleaved kernels; it shows what a planar layout would gain if the data already were
  planar. Add `deinterleave_rgb` to get what this backend pays.
- `pytorch` (eager): five kernels in FP32, three multiplies (one per channel) and two adds.
  Look at DRAM read bytes per kernel: a channel slice `image[..., c]` is strided, so each
  multiply reads every sector of the image.
- `pytorch_matmul`: one cuBLAS GEMV kernel. Look at instructions per pixel, issue slots
  and active lanes per instruction (Source page): a reduction over 3 values leaves most
  lanes idle.

## Generated code

```bash
so=artifacts/torch_extensions/gpu_lab_rgb_to_grayscale/*.so
cuobjdump -sass $so | grep -E "Function|LDG|STG|LDS|STS|FFMA|FMUL|DFMA"
cuobjdump --dump-resource-usage $so                      # registers per kernel
```

Expect three `LDG.E` (`LDG.E.U16` in FP16) and one `STG.E` in `rgb_to_grayscale_pixel`;
three `LDG.E.128` and one `STG.E.128` in the main path of `rgb_to_grayscale_vec`, with
scalar loads only in the tail loop; `LDG.E.128` + `STS.128`, then three `LDS` and one
`STG.E` in `rgb_to_grayscale_smem`. Each pixel needs one `FMUL` and two `FFMA`; any
`DFMA`/`DMUL` would mean a double literal slipped in. Triton's code for
`pytorch_compile` is visible on the Source page of its report: 6 scalar loads
(`LDG.E.EL`) and one `STG.E.64` (FP32) per thread, i.e. 2 pixels per thread.
