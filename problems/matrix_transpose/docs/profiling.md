# Profiling

[← Matrix Transpose](../README.md) · [Design](design.md) · [Results](results.md) · [Testing](testing.md) ·
[Benchmarking](benchmarking.md) · **Profiling**

Lab-wide workflow and `lab.profile`: [profiling](../../../docs/profiling.md).

## Hypotheses under test

1. `cuda_smem` reads tile columns with 32-way (FP32) / 16-way (FP16) bank conflicts;
   `cuda_smem_pad` removes them in FP32 but leaves up to 2-way in FP16; `cuda_smem_swizzle`,
   `cuda_vec_ilp` and Triton's compiler-generated layout conversion are conflict-free. Evidence: shared-load wavefronts per instruction
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
| `ncu/<kernel>_<dtype>_7000x6000_cold.ncu-repz` | 16 | selected metrics, six CUDA kernels + `triton` + the copy `cuda_vec` ceiling (a plain copy), FP32/FP16 |
| `ncu/<kernel>_<dtype>_2048_warm.ncu-repz` | 16 | the same, L2-resident |
| `ncu/full/<kernel>_<dtype>_7000x6000.ncu-repz` | 14 | `--set full` with source counters, six CUDA kernels + `triton`, FP32/FP16 |
| `nsys/timeline.nsys-rep` | 1 | CUDA API, kernels and NVTX ranges, 10 calls per backend and dtype |

Each `.ncu-repz` has a `.csv` raw-page export next to it; the timeline has
`timeline_cuda_gpu_kern_sum.csv` and `timeline_nvtx_sum.csv` (regenerated on every sweep:
`nsys stats` needs `--force-overwrite`, or it keeps the previous run's CSVs). Triton's FP32
and FP16 kernels share the name `transpose_kernel`, so the kernel summary merges them;
split them by NVTX range in `timeline.sqlite`.

## Reading the evidence

- **Bank conflicts per instruction.** In a full report, open Source → SASS and look at the
  `LDS` row: `L1 Wavefronts Shared` against `L1 Wavefronts Shared Ideal`. For `cuda_smem`
  FP32 it is 42.0 M against 1.31 M (32×); for `cuda_smem_pad` FP16 1.97 M against 1.31 M
  (1.5×); for the swizzled tiles they are equal. The CLI equivalent:
  `ncu --import <report> --page source --print-source sass --csv`.
- **Aggregate counters can disagree with the source view.** `cuda_vec_ilp` and `triton`
  report some aggregate shared conflicts while every one of their shared instructions
  shows zero excess wavefronts (see
  [results](results.md#shared-memory-counters-of-cuda_vec_ilp-and-triton)). Trust the
  per-instruction view for address conflicts.
- **Triton's shared ratios need the SASS.** Its layout conversion writes with `STSM`
  (store-matrix), which `smsp__sass_inst_executed_op_shared_st` does not count, so the
  aggregate store wavefronts per instruction reads 0.00. Its loads are `LDS.128`, whose
  ideal is 4 wavefronts per instruction, not 1: the aggregate 4.71 (FP32 cold) is 330,880
  instructions × 4 plus the 235,837 counted conflicts.
- **ncu durations are base-clock, single-launch kernel times.** They are lower-clocked and
  exclude call overhead, so compare kernels with each other within ncu and use the
  benchmark (or the Nsight Systems timeline, at unlocked clocks) for absolute latency.
  Clock locking magnifies instruction-bound differences: FP16 `cuda_vec_ilp` takes 11% less
  time than the padded tile under ncu (194.2 vs 218.0 µs) but only 2% less per call.
- **Single cold captures vary by up to 8%.** Repeating the same capture three times gave
  393, 423 and 393 µs for `cuda_smem_swizzle` FP32 (168.3, 182.6, 168.0 MB read), 393, 394
  and 409 µs for `cuda_smem_pad`, and 389, 424 and 393 µs for `triton`. The slow captures
  read up to 8.7% more DRAM. Repeat a capture before reading a few-percent difference into
  it.
- **DRAM writes look short.** Cold captures report 136–141 MB written for 168 MB of output:
  the remaining dirty lines are still in L2 when the kernel ends.
- **Sectors per request above the ideal on 7000 × 6000.** An FP16 output row is 14,000
  bytes, so every other row starts mid-sector: coalesced 64-byte FP16 stores average 2.5
  sectors, and `cuda_vec_ilp`'s 32-byte pack segments 20.9 instead of 16. Its edge tiles
  (6000 is not a multiple of 256) use scalar accesses, giving 15.0 rather than 16 in FP32.
