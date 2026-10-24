# Profiling

[← Interleave Arrays](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · **Profiling**

Lab-wide workflow and `lab.profile`: [profiling](../../../docs/profiling.md). The same
four-question analysis is explained in more depth for
[Reverse Array](../../reverse_array/docs/profiling.md).

## Capturing one launch

```bash
ncu --profile-from-start off --set full --section SourceCounters --import-source yes \
  -o artifacts/interleave_arrays/ncu/output_fp16_25000000 \
  python -m lab.profile interleave_arrays cuda_output fp16 n=25000000
```

`lab.profile` checks the backend against the reference, warms it up and runs it once in a
capture range, so input generation never appears. The PyTorch reference produces two
kernels per report (one per strided assignment). [sweep.sh](../benchmarks/sweep.sh)
originally captured the 10 CUDA/PyTorch reports used in the historical analysis
in [results](results.md). It now also includes two Triton reports when run without
`BENCH_ONLY=1`; the current Triton follow-up collected benchmarks only.

## What to check, in order

| Question | Where | Metric | Healthy value here |
|---|---|---|---|
| Extra DRAM traffic? | Memory Workload Analysis; Raw page | `dram__bytes_op_read.sum`, `dram__bytes_op_write.sum` | read = `2·N·s`, write ≤ `2·N·s` |
| Sectors completed in L2? | Raw page | `lts__t_sector_op_write_hit_rate.pct` | > 0 when two stores share a sector (the halves meeting in L2) |
| Wasteful requests? | Memory Workload Analysis → L1/TEX table; Source page | Global load/store **Sectors/Req**; per instruction **L2 Theoretical Sectors Global** vs **…Ideal** | equal to ideal, or wasted sectors that L2 absorbs |
| DRAM kept busy? | GPU Speed Of Light | **DRAM Throughput** (`gpu__dram_throughput…`) | ~88–89%, like a copy |
| Half-empty warps? | Source page | **Avg. Predicated-On Threads Executed** per memory instruction | 32 |
| Why warps wait | Warp State Statistics | stall reasons: long scoreboard (memory), LG throttle (memory-instruction queue full) | long scoreboard dominates |
| Overhead? | Instruction Statistics, Occupancy | instructions per element, registers, warps active | irrelevant while DRAM-bound |

The two findings in [results](results.md#why-the-variants-differ) each come from one row:
PyTorch's slowdown shows up only in DRAM **read bytes** (its per-instruction pattern is
identical to `cuda_input`'s), and `cuda_output` FP16's shows up only in **active lanes**
and DRAM busy (its byte counts are minimal). Neither is visible from sectors per request
alone.

## Generated code

```bash
so=artifacts/torch_extensions/gpu_lab_interleave_arrays/*.so
cuobjdump -sass $so | grep -E "Function|LDG|STG"   # access widths, predication (@P0)
cuobjdump --dump-resource-usage $so               # registers per kernel
```

Expect two scalar loads and two stride-2 stores in `interleave_input`; predicated loads
(`@P0` / `@!P0`) in `interleave_output`, plus split predicated stores in FP16;
`STG.E.64` (FP32) or a packed `STG.E` (FP16) in `interleave_pair`; and two `LDG.E.128` and
two `STG.E.128` in `interleave_vec`.
