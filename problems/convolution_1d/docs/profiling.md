# Profiling

[← 1D Convolution](../README.md) · [Design](design.md) · [Results](results.md) · [Testing](testing.md) ·
[Benchmarking](benchmarking.md) · **Profiling**

Every report is one launch captured by [capture.py](../benchmarks/capture.py), after the
backend has been checked against the reference, compiled and warmed up, with caches flushed
(`--cache-control all`). Lab-wide guidance is in the [profiling plan](../../../docs/profiling.md).

```bash
ncu --profile-from-start off --cache-control all --metrics "$metrics" \
  -o artifacts/convolution_1d/ncu/cuda_smem_reg_fp32_k255 \
  .venv/bin/python -m problems.convolution_1d.benchmarks.capture cuda_smem_reg fp32 16777216 255
```

`$metrics` is the list in [sweep.sh](../benchmarks/sweep.sh). `--reg-outputs R` selects
`cuda_smem_reg`'s outputs per thread; `copy_cuda_vec` captures 07's copy kernel.

## Reports

| Reports | Count | Contents |
|---|---:|---|
| `ncu/<backend>_<dtype>_k<K>` | 52 | Every CUDA kernel and Triton, FP32 and FP16, K = 3, 31, 255, 2047, N = 2²⁴ (`cuda_unrolled` at K = 3 and 31 only) |
| `ncu/cuda_smem_reg_r<R>_<dtype>_k255` | 8 | R = 2, 4, 8, 16: registers, occupancy and FMA utilization against R |
| `ncu/copy_cuda_vec_<dtype>` | 2 | 07's copy at N = 2²⁴: the bandwidth control |
| `halo/halo_r<R>_k<K>` | 16 | `cuda_large_k`, FP32, R = 2–16 (256·R outputs per block), K = 31, 255, 1023, 2047: global-load sectors for the halo model |

Each `.ncu-rep` has a raw-page `.csv` export next to it, which
[summarize.py](../benchmarks/summarize.py) reads.

## Metrics and the question each answers

| Metric | Question |
|---|---|
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` per warp-FMA | Shared loads per FMA: the direct test of `cuda_smem` (≈ 1) against `cuda_smem_reg` (≈ 1/R) |
| `l1tex__lsuin_requests.sum` per warp-FMA, `l1tex__throughput` | LSU/L1 requests per FMA and how saturated L1 is (the naive kernels' limiter, [design §4](design.md#4-triton-the-naive-stencil-measured)) |
| `sm__pipe_fma_cycles_active`, `smsp__inst_executed_pipe_fma`, `smsp__inst_executed` | How close each variant gets to compute-bound, and its instruction overhead per FMA |
| `l1tex__t_sector_hit_rate.pct`, `dram__bytes_op_read/write.sum` | L1 absorbing the K-fold reuse (naive against smem), and DRAM bytes per output against the 2·e minimum |
| `smsp__average_warps_issue_stalled_{barrier,lg_throttle,mio_throttle,long_scoreboard,math_pipe_throttle}` | Barrier cost of the shared tiles at small K; which pipe the warps wait on |
| `launch__registers_per_thread`, `sm__maximum_warps_per_active_cycle_pct`, `sm__warps_active` | Register pressure against R: theoretical and achieved occupancy |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` (halo reports) | Global elements loaded per output against the model chunks + 2K/span |

## Caveats

- **ncu durations are single-launch kernel times** at whatever clock the GPU runs under the
  profiler (recorded as `sm__cycles_elapsed.avg.per_second`); compare kernels with each other
  within ncu and use the benchmark tables for per-call latency.
- **Per-FMA ratios use the useful FMA count**, (N − K + 1)·K / 32 warp-FMAs, not executed
  FFMA instructions, so a kernel that wastes FMAs on padding would look worse, not better.
- Triton's kernel is profiled with its default tile (256 outputs × 4 warps in FP32, 512 in
  FP16).
