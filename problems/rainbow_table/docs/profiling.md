# Profiling: what to inspect and why

[← Rainbow Table](../README.md) · [Results](results.md) · [SASS](sass.md) · [Roofline](roofline.md)

## Open the captures

Reports live in `artifacts/rainbow_table/ncu/` locally and under the same relative
path on cuda.0, in `/home/vdeviatkov/Documents/hpc-gpu-kernel-lab`. Open
`cuda_u32_r1.ncu-repz` in Nsight Compute, add `cuda_u32_r100.ncu-repz` as a baseline,
then compare `cuda_ilp2`, `cuda_ilp`, `cuda_vec`, `cuda_unroll`, `cuda_int64`, Triton
and compiled PyTorch. R = 10 reports use source N = 5M; R = 1/100 use N = 10M.
That size difference is deliberate: compare endpoints to study R at fixed N, and
use R = 10 separately to explain the source result.

```bash
ncu --profile-from-start off --set full --section SourceCounters --import-source yes \
  --cache-control all -o artifacts/rainbow_table/ncu/example \
  python -m lab.profile rainbow_table cuda_ilp int32 n=10000000 rounds=100 dist=source
ncu --import artifacts/rainbow_table/ncu/example.ncu-repz --page raw --csv
ncu --import artifacts/rainbow_table/ncu/example.ncu-repz --page source --print-source sass
```

The runner checks exact correctness, compiles and warms up before the capture.
Profiles flush caches; benchmark inputs are reused. Nsight replay, base clock
control and cache flushing change timings: compare within the same measurement
method. Eager `pytorch` / `pytorch_int32` reports intentionally capture only the
first launch (`--launch-count 1`); they cannot represent the full eager backend or
its total traffic. Benchmark JSON times the entire call. CUDA aliases are captured
separately as a reproducibility control.

## Inspection order and definitions

| Where / exact metric | Definition | Check and reasoning |
|---|---|---|
| Speed of Light: `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Maximum utilization among measured SM throughput constituents, normalized to sustained peak over elapsed time | Rising with R suggests compute pressure. This is not the fraction of SMs doing useful hash work. |
| Speed of Light: `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM unit throughput as a percent of sustained peak | High at small R would support a DRAM limit; low values require checking L2, latency and issue pressure. |
| Memory: `dram__bytes_op_read.sum`, `dram__bytes_op_write.sum` | Actual DRAM bytes during the profiled kernel | Compare with logical 4N read + 4N write; cache hits and deferred writes change physical traffic. This version reports these byte counters with SI units in CSV; retain those units. |
| Memory: `lts__throughput.avg.pct_of_peak_sustained_elapsed`, `lts__t_sector_hit_rate.pct` | Busiest L2 constituent throughput; fraction of sectors hitting L2 | High L2 and low DRAM means a cache ceiling. The source's 40 MB logical working set is L2-sized. |
| Compute: `sm__inst_executed_pipe_alu.sum`, `sm__inst_executed_pipe_fma.sum`, `sm__inst_executed_pipe_aluheavy.sum`, `sm__inst_executed_pipe_fmaheavy.sum`, `sm__inst_executed_pipe_fmalite.sum` | Warp instructions executed by each named pipeline | Normalize by NR and inspect SASS. A pipeline named FMA can execute integer instructions; do not infer FP work from its name. Parent and subpipe counters can overlap; do not add all of them. |
| Scheduler: `smsp__issue_active.avg.pct_of_peak_sustained_active` | Fraction of active scheduler cycles issuing an instruction | High issue with integer-pipe saturation supports throughput limitation. Low issue with low eligible warps points toward dependencies. Active-cycle normalization differs from elapsed-cycle SOL. |
| Scheduler: `smsp__warps_eligible.avg.per_cycle_active` | Warps ready to issue per scheduler active cycle | More than one gives scheduler choice; near zero means it often has nothing ready. Compare ILP with scalar at R = 100. |
| Warp State: `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | Waiting warp cycles for fixed-latency dependencies, per issued instruction | Large values support serial math dependency chains. ILP should expose independent work, but confirm with time and eligible warps. |
| Warp State: `smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio` | Warp cycles waiting for an execution pipe, per issued instruction | Large values plus a busy pipe indicate competition for that pipe; extra ILP may already have filled it. |
| Warp State: `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | Warp cycles waiting for L1TEX scoreboard dependencies, per issued instruction | Typically load latency, including local/global/texture operations. Large values alone do not prove bandwidth saturation. |
| Occupancy: `sm__warps_active.avg.pct_of_peak_sustained_active` | Resident active warps relative to hardware capacity | Residency is not readiness. A higher value can still lose if each warp has only a serial dependency chain. |
| Launch: `launch__registers_per_thread`; Occupancy limits | Allocated registers/thread; block residency constraints | Check whether ILP changes resident blocks. Registers can increase without reducing occupancy because allocation is granular and other limits apply. |
| Source: instructions, active lanes, sampled stalls | SASS operation and per-instruction execution/sample evidence | Find prime IMADs, fused LOP3s and SHFs; locate dependency producer/consumer pairs. Sampling can be sparse for short R = 1 kernels. |

Stall ratios in this table count waiting warp cycles **per issued instruction**;
they are not percentages and can exceed one. Nsight also shows PC-sampling stall
percentages: keep their denominators separate. A warp stall does not necessarily
stall the SM because another warp may issue. NVIDIA's
[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) defines
these states and explains normalization and sampling limitations.

## Questions to answer from the pair

1. Does R = 1 saturate DRAM, L2, issue, or none? Confirm with bytes and utilization.
2. At R = 100, do prime multiplies account for approximately `4NR/32` full-warp
   issues (adjust for K and tails)? Which measured pipe contains this work?
3. Does the scalar kernel show wait stalls despite high occupancy? Does ILP reduce
   wait or increase eligible warps, or does it only shift the limit to math throttle?
4. Is unroll's benefit due to eliminating uniform loop control, and does its extra
   LOP3 / larger code offset that benefit? Inspect instruction footprint too.
5. For int64, is there actually a multi-instruction 64-bit multiply? The CUDA
   compiler eliminated it here; verify Source rather than trusting source types.
6. Compare the same NR, cache policy and clocks. Use the custom integer roofline;
   the standard FP roofline does not measure this workload's work.
