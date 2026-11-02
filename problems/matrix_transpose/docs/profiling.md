# Profiling

[← Matrix Transpose](../README.md) · [Design](design.md) · [Results](results.md) · [Testing](testing.md) ·
[Benchmarking](benchmarking.md) · **Profiling**

Lab-wide workflow and `lab.profile`: [profiling](../../../docs/profiling.md).

## Hypotheses under test

1. `cuda_smem` reads tile columns with 32-way (FP32) / 16-way (FP16) bank conflicts;
   `cuda_smem_pad` removes them in FP32 but leaves up to 2-way in FP16; `cuda_smem_swizzle`
   and `cuda_vec_ilp` are conflict-free. Evidence: shared-load wavefronts per instruction
   and `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum`.
2. The naive kernels coalesce one side only. Evidence: global sectors per request
   (`l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}` over `..._t_requests_...`): 4 (FP32)
   or 2 (FP16) when coalesced, 32 when every lane touches its own sector.
3. Every tiled kernel moves the minimum DRAM traffic. Evidence: `dram__bytes_op_read.sum`.

All three are confirmed; see [results](results.md).

## Captures

```bash
# Cold: the source's 7000 x 6000, caches flushed before the launch (DRAM-bound)
ncu --profile-from-start off --metrics "$metrics" --cache-control all \
  -o artifacts/matrix_transpose/ncu/cuda_smem_fp32_7000x6000_cold \
  python -m lab.profile matrix_transpose cuda_smem fp32 m=7000 n=6000

# Warm: 2048², caches kept, so the warmups leave the matrix in L2 as in the benchmark loop
ncu --profile-from-start off --metrics "$metrics" --cache-control none \
  -o artifacts/matrix_transpose/ncu/cuda_smem_fp32_2048_warm \
  python -m lab.profile matrix_transpose cuda_smem fp32 m=2048 n=2048

# Every section, with SASS-level source counters, for inspection in the GUI
ncu --profile-from-start off --cache-control all --set full --import-source yes \
  -o artifacts/matrix_transpose/ncu/full/cuda_smem_fp32_7000x6000 \
  python -m lab.profile matrix_transpose cuda_smem fp32 m=7000 n=6000

# Timeline: every kernel and both copy controls on 7000 x 6000, one NVTX range each
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o artifacts/matrix_transpose/nsys/timeline \
  python -m problems.matrix_transpose.benchmarks.timeline
```

`$metrics` is the list in [sweep.sh](../benchmarks/sweep.sh). `lab.profile` checks the
result against the reference and warms up before `cudaProfilerStart`, so a capture holds
exactly one launch of the kernel under test.

| Report set (`artifacts/matrix_transpose/`) | Count | Contents |
|---|---:|---|
| `ncu/<kernel>_<dtype>_7000x6000_cold.ncu-repz` | 14 | selected metrics, six kernels + the copy `cuda_vec` ceiling (a plain copy), FP32/FP16 |
| `ncu/<kernel>_<dtype>_2048_warm.ncu-repz` | 14 | the same, L2-resident |
| `ncu/full/<kernel>_<dtype>_7000x6000.ncu-repz` | 12 | `--set full` with source counters, six kernels, FP32/FP16 |
| `nsys/timeline.nsys-rep` | 1 | CUDA API, kernels and NVTX ranges, 10 calls per backend and dtype |

Each `.ncu-repz` has a `.csv` raw-page export next to it; the timeline has
`timeline_cuda_gpu_kern_sum.csv` and `timeline_nvtx_sum.csv`.

## Reading the evidence

- **Bank conflicts per instruction.** In a full report, open Source → SASS and look at the
  `LDS` row: `L1 Wavefronts Shared` against `L1 Wavefronts Shared Ideal`. For `cuda_smem`
  FP32 it is 42.0 M against 1.31 M (32×); for `cuda_smem_pad` FP16 1.97 M against 1.31 M
  (1.5×); for the swizzled tiles they are equal. The CLI equivalent:
  `ncu --import <report> --page source --print-source sass --csv`.
- **Aggregate counters can disagree with the source view.** `cuda_vec_ilp` reports some
  aggregate shared conflicts while every one of its `LDS`/`STS` instructions shows zero
  excess wavefronts (see [results](results.md#shared-memory-counters-of-cuda_vec_ilp)).
  Trust the per-instruction view for address conflicts.
- **ncu durations are base-clock, single-launch kernel times.** They are lower-clocked and
  exclude call overhead, so compare kernels with each other within ncu and use the
  benchmark (or the Nsight Systems timeline, at unlocked clocks) for absolute latency.
  Clock locking magnifies instruction-bound differences: FP16 `cuda_vec_ilp` takes 15% less
  time than the padded tile under ncu (182.8 vs 215.8 µs) but only 1.5% less per call.
- **DRAM writes look short.** Cold captures report 140 MB written for 168 MB of output: the
  remaining dirty lines are still in L2 when the kernel ends.
- **Sectors per request above the ideal on 7000 × 6000.** An FP16 output row is 14,000
  bytes, so every other row starts mid-sector: coalesced 64-byte FP16 stores average 2.5
  sectors, and `cuda_vec_ilp`'s 32-byte pack segments 20.9 instead of 16. Its edge tiles
  (6000 is not a multiple of 256) use scalar accesses, giving 15.0 rather than 16 in FP32.
