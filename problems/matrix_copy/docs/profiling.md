# Profiling

[← Matrix Copy](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · **Profiling**

Lab-wide workflow and `lab.profile`: [profiling](../../../docs/profiling.md).

## Capturing one launch

```bash
# Cold: the source's 4096², caches flushed before the launch (DRAM-bound)
ncu --profile-from-start off --metrics "$metrics" --cache-control all \
  -o artifacts/matrix_copy/ncu/cuda_vec_fp32_4096_cold \
  python -m lab.profile matrix_copy cuda_vec fp32 m=4096 n=4096

# Warm: 2048², caches kept, so the warmups leave the matrix in L2 as in the benchmark loop
ncu --profile-from-start off --metrics "$metrics" --cache-control none \
  -o artifacts/matrix_copy/ncu/cuda_vec_fp32_2048_warm \
  python -m lab.profile matrix_copy cuda_vec fp32 m=2048 n=2048
```

`$metrics` is the list in [sweep.sh](../benchmarks/sweep.sh). The sweep captures each of the
ten kernels in FP32 and FP16, cold and warm: 40 reports. `pytorch` is not profiled: its
copy is a `cudaMemcpyAsync`, which ncu's kernel profiler does not capture.

ncu locks clocks to base and times the kernel alone, so its durations are lower-clocked
and exclude the call overhead in the benchmark. Compare kernels with each other within
ncu, and use the benchmark for absolute call latency.

**Warm captures only work well below L2 size.** ncu replays a kernel once per metric pass
and saves and restores device memory around each pass, which evicts L2. At 2048² the
matrix is small enough to survive; near the L2 capacity it is not, and even a
steady-state capture shows ~0.1% read hits. Also, `lab.profile` drops each output at once
(one reused output block), while `lab.bench` keeps two alive. To reproduce the
benchmark's state, use [steady_state.py](../benchmarks/steady_state.py), which times both
patterns and with `--profile` captures two calls of the benchmark-style loop.

## What to check, in order

### 1. Bytes: does the kernel move the minimum?

| Metric | Meaning | Expected |
|---|---|---|
| `dram__bytes_op_read.sum` | Bytes read from DRAM | `N²·e` cold (67.1 MB at 4096² FP32); ~0 warm |
| `dram__bytes_op_write.sum` | Bytes written to DRAM *during* the kernel | ≤ `N²·e`: dirty lines still in L2 at kernel end are written later |
| `lts__t_sector_hit_rate.pct` | L2 sector hit rate | Low cold; ~100% warm |

**Decision:** DRAM reads above `N²·e` mean sectors were fetched twice. A copy has no
reuse, so this would only happen if L2 evicted a sector before all of its bytes were used.

### 2. Coalescing: sectors per request

| Metric | Meaning | Expected |
|---|---|---|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum` | 32-byte sectors touched by global loads / stores | `N²·e / 32` for coalesced kernels |
| `l1tex__t_requests_pipe_lsu_mem_global_op_{ld,st}.sum` | Warp-level load / store instructions | |
| Sectors / request (their ratio) | How spread out one warp's access is | 4 for 4-byte scalar, 2 for 2-byte, 16 for 16-byte packs; **32** for the column-major kernel |

**Decision:** a high ratio is only harmful if the extra sectors are not reused. In
`cuda_2d_colmajor` a warp touches 32 rows, but the block's 8 warps cover 8 adjacent
columns, which for FP32 is exactly one sector per row. Whether L1/L2 serve the other seven
touches (L2 hit rate, step 1) decides how much the pattern costs.

### 3. Bandwidth: is DRAM (or L2) kept busy?

| Metric | Meaning |
|---|---|
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM bandwidth used over the kernel, % of peak: the cold-regime limit |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 throughput: the warm-regime limit |
| `gpu__time_duration.sum` | Kernel time at ncu's locked clocks |

### 4. Latency hiding: occupancy and stalls

| Metric | Meaning |
|---|---|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy |
| `launch__registers_per_thread`, `launch__grid_size`, `launch__block_size` | What was launched |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | Waiting on global memory: expected to dominate |
| `smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio` | Load/store queue full: many narrow requests |
| `smsp__inst_executed.sum` | Warp instructions: what vectorization saves |

**Decision:** in the cold regime, any kernel that keeps DRAM ~90% busy is done, whatever
its occupancy or instruction count. In the warm regime, the limit becomes L2 and request
issue, so fewer and wider requests (fewer instructions, less LG throttle) should win.

## SASS

`sweep.sh` dumps the CUDA extension with `cuobjdump --dump-sass` and the Triton kernels from
their compiled `asm["sass"]`; [inspect_sass.py](../benchmarks/inspect_sass.py) counts
global loads and stores by width and qualifier. Check that every pack variant uses
`LDG.E.128` / `STG.E.128`, and what Triton emits for its 1-D and 2-D layouts.
