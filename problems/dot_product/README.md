# 16 · Dot Product

## Source

LeetGPU: [Dot Product](https://leetgpu.com/challenges/dot-product)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[13 · Reduction](../reduction/README.md); [15 · Mean Squared Error](../mean_squared_error/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

A two-input reduction is the immediate conceptual bridge to GEMV and supplies an experiment in accumulator dependency chains.

## Primary lesson

**Fused multiply-accumulate reduction**

## Primary concepts

- Reductions
- Instruction mix
- Dependency chains
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
dot_product/
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

1. Multiply then reduce baseline.
2. Fuse multiplication into hierarchical reduction.
3. Compare single and multiple accumulators per thread.
4. Study FMA and mixed-precision lab variants.

## Correctness focus

Cancellation, odd lengths, extreme products, and explicit FMA/precision policy.

## Things to measure

- Latency
- Effective GB/s
- Useful FLOP/s
- Register pressure and error

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Do independent accumulators increase useful instruction throughput?
- Are two input streams limited by memory or dependency latency?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
