# Profiling

[← Reverse Array](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · **Profiling**

Lab-wide workflow and `lab.profile`: [profiling](../../../docs/profiling.md).

## Capturing one launch

[lab/profile.py](../../../lab/profile.py) checks the backend against the reference, warms
it up, and runs it once inside a profiler capture range, so input generation never
appears in a report:

```bash
ncu --profile-from-start off --set full --section SourceCounters --import-source yes \
  -o artifacts/reverse_array/ncu/tile_fp16_100000001 \
  python -m lab.profile reverse_array cuda_tile fp16 n=100000001
```

`--set full` collects every section; `SourceCounters` with `--import-source yes` adds
per-instruction counters linked to the CUDA source (the extensions build with
`-lineinfo`). [sweep.sh](../benchmarks/sweep.sh) captures all 20 reports used in
[results](results.md). Open them in the Nsight Compute GUI; to compare two, open one and
use **Add Baseline** on the other.

## How the analysis is organized

A memory-bound kernel can lose to a copy in four ways, and each has a place in the
report:

| Question | Report section | Metric | Healthy value here |
|---|---|---|---|
| Moving more bytes than needed? | Memory Workload Analysis → L1/TEX table | Global load/store **Sectors/Req** | 4 (FP32 scalar), 2 (FP16 scalar), 16 (16-byte) |
| Per instruction, which access wastes sectors? | Source page, SASS view | **L2 Theoretical Sectors Global** vs **…Ideal** | equal |
| DRAM kept busy? | GPU Speed Of Light | **DRAM Throughput** (= Memory Workload's **Max Bandwidth**) | ~87%, like a copy |
| Why are warps waiting? | Warp State Statistics | stall reasons (long scoreboard = memory, barrier, MIO throttle) | memory dominates |
| Another unit saturated? | Memory Workload Analysis → Shared Memory; **Mem Busy** | bank conflicts, shared pipeline % | pipeline well below 80% |
| Overhead between similar variants? | Instruction Statistics, Occupancy | instructions per element, block limits | similar |

Two numbers in Memory Workload Analysis are easy to confuse. **Max Bandwidth** is how
full the busiest data link is (for this kernel, the DRAM link, ~87%). **Mem Busy** is how
hard the memory units themselves work (~18% here): they mostly pass misses through. High
bandwidth with low Mem Busy is the expected state of a memory-bound kernel.

A kernel-wide sectors-per-request value can hide uncoalesced instructions. In
`triton_pair` each thread loads 4 consecutive FP32 elements of the back half with 4
scalar instructions, so within one instruction the 32 lanes are 16 bytes apart: the
request spans 512 bytes, which is 16 sectors, exactly the count of a perfect 128-bit
load. Sectors/request alone cannot tell them apart; the per-instruction **Ideal** column
can (1 used sector per 4 touched). Check the per-instruction view before concluding.

## Generated code

```bash
so=artifacts/torch_extensions/gpu_lab_reverse_array/*.so
cuobjdump -sass $so | grep -E "Function|LDG|STG|LDS|STS|LDGSTS|BAR"   # access widths
cuobjdump --dump-resource-usage $so                                  # registers per kernel
```

Expect `LDG.E.128`/`STG.E.128` in `reverse_vec` and the tile kernels, `LDGSTS.E.BYPASS.128`
plus `LDGDEPBAR`/`DEPBAR` in `reverse_tile_async`, and scalar accesses in `reverse_pair`.
For Triton, a launch returns a compiled kernel whose `asm["sass"]` holds the same
listing.
