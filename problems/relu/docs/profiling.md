# Profiling

[← ReLU](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · **Profiling**

Lab-wide workflow: [profiling](../../../docs/profiling.md).

## Capturing exactly one ReLU launch

[profile.py](../benchmarks/profile.py) builds the inputs, compiles and warms up the
backend, then calls it once between `cudaProfilerStart` and `cudaProfilerStop`. Profile
with the capture range so input generation (which launches its own PyTorch kernels) is
never captured:

```bash
ncu --profile-from-start off --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum,\
smsp__sass_inst_executed_op_global_st.sum,smsp__sass_branch_targets_threads_divergent.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread \
  python -m problems.relu.benchmarks.profile cuda_branch fp16 25000000 alternating
```

Arguments: backend, dtype, N, sign distribution. The kernel name in the output confirms
what was captured (`relu_select<float>`, `triton relu_kernel`, or PyTorch's
`vectorized_elementwise_kernel`).

| Metric | What it shows here |
|---|---|
| `dram__throughput…pct_of_peak_sustained_elapsed` | whether memory is saturated |
| `smsp__inst_executed.sum` / N | instructions per element |
| `smsp__sass_inst_executed_op_global_st.sum` / (N / 32) | store instructions per warp |
| `smsp__sass_branch_targets_threads_divergent.sum` | divergent branches (0 = no branching on data) |
| `sm__warps_active…` | achieved occupancy |

For launch and host overhead at small sizes, use Nsight Systems with the same script:

```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -o artifacts/relu_nsys \
  python -m problems.relu.benchmarks.profile cuda_select fp32 1025 mixed
```

## Reading the SASS

```bash
so=artifacts/torch_extensions/gpu_lab_relu/*.so
cuobjdump -sass $so | grep -E "Function|FSEL|FMNMX|FSETP|BRA|EXIT|LDG|STG"
```

What to look for: `FSETP` + `FSEL` is a branch-free select; `FMNMX` is `fmaxf`;
predicated `@!P0 STG` / `@!P0 EXIT` is a predicated `if`. Every function ends with
`BRA` to its own address after the last `EXIT`; that instruction never runs and is not
a branch on data.
