# 21 · Dense GEMV

## Source

**Portfolio Extension** — independently scoped laboratory workload, not a LeetGPU challenge.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[08 · Matrix Transpose](../matrix_transpose/README.md); [16 · Dot Product](../dot_product/README.md)

## Difficulty

- LeetGPU: **Not applicable — Portfolio Extension**.

## Why this problem matters

A deliberate portfolio extension fills the gap between dot product and GEMM without mislabeling sparse matrix-vector multiplication as dense GEMV.

## Primary lesson

**Low-arithmetic-intensity matrix math**

## Primary concepts

- Reductions
- Memory coalescing
- Arithmetic intensity
- Data layouts

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
dense_gemv/
    README.md
    cuda/
    triton/
    pytorch/
    jax/
    tests/
    benchmarks/
    figures/
    docs/
    results/
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Define row-major y = A x with FP32 accumulation.
2. One cooperative row per block or warp.
3. Compare transposed-storage mappings as separate contracts.
4. Explore split reduction for very wide rows.

## Correctness focus

Tall/wide matrices, odd dimensions, FP16/BF16 lab inputs, and explicit output precision.

## Things to measure

- Latency
- Effective GB/s
- FLOP/s
- Vector-cache reuse

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Which aspect ratios expose too little parallelism?
- How much reuse of x reaches L2 or registers?
- Does split reduction repay its extra pass?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
