# 10 · Gaussian Blur

## Source

LeetGPU: [Gaussian Blur](https://leetgpu.com/challenges/gaussian-blur)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[09 · 1D Convolution](../convolution_1d/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

This is retained as a distinct algorithmic experiment: separability can save arithmetic while introducing an intermediate image and another launch.

## Primary lesson

**Separable filtering tradeoffs**

## Primary concepts

- Tiled memory access
- Shared memory
- Recomputation vs traffic

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
gaussian_blur/
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

1. Direct two-dimensional filter with exact boundary semantics.
2. Shared image tile and halo.
3. Use two separable passes only when the supplied kernel is separable.
4. Compare intermediate traffic against saved arithmetic.

## Correctness focus

Image edges, rectangular kernels, normalization, and numerical effects of factorization.

## Things to measure

- Latency
- Pixels/s
- Logical and measured traffic
- Launch count

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is the supplied filter actually separable within the required tolerance?
- When do two passes lose to a fused direct tile?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
