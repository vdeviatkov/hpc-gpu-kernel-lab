# Profiling

[← Vector Addition](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · **Profiling**

Lab-wide workflow: [profiling](../../../docs/profiling.md).

`benchmarks.run --profile` builds and warms up before opening a CUDA profiler capture
range, so build and JIT never enter the capture. Choose exactly one backend, size and
dtype; profile runs produce no benchmark JSON.

## Nsight Compute: kernel-level hypotheses

```bash
mkdir -p artifacts
ncu --profile-from-start off --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section LaunchStats --launch-count 1 -o artifacts/vector_add_ncu \
  python -m problems.vector_add.benchmarks.run --profile --backends cuda_vec \
  --sizes 25000000 --dtypes fp16 --samples 1
```

The [2×2 table in results](results.md#what-the-22-design-shows) uses these metrics:

```bash
ncu --profile-from-start off --launch-count 1 --csv --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,\
launch__registers_per_thread,launch__grid_size \
  python -m problems.vector_add.benchmarks.run --profile --backends cuda_scalar \
  --sizes 25000000 --dtypes fp32 --samples 1
```

Replay and cache-control policies change what the profiler observes, so keep the
profiler configuration with its results and time normal runs separately.

## Nsight Systems: launch and dispatch costs

```bash
nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o artifacts/vector_add_nsys \
  python -m problems.vector_add.benchmarks.run --profile --backends cuda_scalar \
  --sizes 1025 --dtypes fp32 --samples 100
```

## Registers and instruction widths

The built extension can be inspected without running it:

```bash
so=artifacts/torch_extensions/gpu_lab_vector_add/*.so
cuobjdump --dump-resource-usage $so          # registers per kernel and dtype
cuobjdump -sass $so | grep -E 'LDG|STG'     # expect LDG.E.128 / STG.E.128 in vec kernels
```

`.CONSTANT` on loads confirms the read-only path enabled by `__restrict__`.
