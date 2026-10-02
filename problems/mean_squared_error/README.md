# 15 · Mean Squared Error

## Source

LeetGPU: [Mean Squared Error](https://leetgpu.com/challenges/mean-squared-error)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[13 · Reduction](../reduction/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

MSE combines arithmetic and reduction, making the cost of materialized differences visible before more complex ML losses.

## Primary lesson

**Map-reduce fusion**

## Primary concepts

- Reductions
- Kernel fusion
- Numerical precision

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
mean_squared_error/
    README.md
    cuda/
    triton/
    pytorch/
    jax/
    tests/
    benchmarks/
    results/
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Separate difference, square, and mean baseline.
2. Fuse the map into a block reduction.
3. Compare one final scaling step with per-element scaling.
4. Study accumulation precision.

## Correctness focus

Identical vectors, large/small differences, empty-input policy, and overflow-sensitive data.

## Things to measure

- Latency
- Intermediate bytes avoided
- Kernel count
- Absolute and relative error

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Does fusion remove traffic or merely move the bottleneck to reduction?
- Where should normalization occur for stable arithmetic?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
