# Profiling plan

Each study should choose measurements from its optimization hypothesis, rather
than collecting every counter and searching for a favorable explanation.

## Nsight Systems: where time goes

Use the timeline to inspect CUDA API calls, kernel launches, stream ordering,
copies, synchronization, allocations, and gaps. It should answer whether a small
workload is launch-bound, whether a reduction has unnecessary passes, or whether
an apparent kernel improvement is hidden by host overhead. Add named ranges
around the measured operation when a real harness exists.

Compile, establish correctness, and warm up before capturing the steady-state
region. Analyze initialization separately when deployment latency is relevant.
Compare a complete MLP/attention pipeline as well as individual kernels. Refer
to the [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
when implementing capture controls.

## Nsight Compute: test the hardware hypothesis

Begin with a small set such as launch/resource information and a throughput
overview. Add memory, scheduler, source, or instruction sections only for the
question at hand. Metric/section availability varies by GPU and tool version;
inspect `ncu --list-sections` and `ncu --query-metrics` on the target host.
The [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
documents section meanings and collection overhead.

| Hypothesis | Evidence to inspect | Relevant curriculum |
|---|---|---|
| DRAM bandwidth limits execution | DRAM bytes/throughput, working-set size, L2 hit rate | Copy, transpose, GEMV, cache reads |
| Accesses waste transactions | Sectors/requests and useful bytes per request | Transpose, RGB, sparse gather |
| Shared-memory layout conflicts | Bank conflicts, shared transactions, tile layout | Padded transpose, histogram, GEMM |
| Synchronization limits issue | Barrier stalls, eligible warps, issue activity | Reduction, scan, normalization |
| Registers constrain residency | Registers/thread, local-memory traffic, achieved/theoretical occupancy | Wide softmax, GEMM tiles, fusion |
| Integer/exponential pipeline is saturated | Instruction mix and pipeline utilization | Hashing, SiLU, softmax |
| Tensor Cores are underused | Matrix-instruction activity, operand staging, tile/padding efficiency | FP16/INT8 GEMM, online attention |
| Work is imbalanced | Per-block work distribution, active versus eligible warps, tail waves | Sparse rows, batched GEMM, MoE |
| Cache reuse explains a speedup | L2 throughput/hits and measured memory bytes | GQA, weight scales, paged caches |

Treat stalls, occupancy, and cache hit rates as diagnostic evidence. A counter
change alone does not establish causality; pair it with the controlled code
change and repeated latency observations. Profiler replay can perturb execution
and memory state, so keep uninstrumented timing separate.

## Capture plan for a flagship

Select one small, one typical, and one resource-stressing shape. Capture baseline
and the proposed variant under the same environment. Record the exact command,
source revision, GPU/tool versions, section selection, expected bottleneck, and
observed differences, and summarize them in the study's `docs/results.md`.

Use memory, race, and synchronization checking before trusting a speedup from
shared memory, in-place updates, or asynchronous work. Those checks are future
validation steps, not profiler output already obtained.

## Evidence placeholder

Status: **Results pending hardware benchmark**.

Future notes should contain the hypothesis, selected metrics and why, measured
observations, alternative explanations, and next experiment. Leave observations
empty until a real capture exists. Do not paste fabricated screenshots or numbers.
