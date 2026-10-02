# 05 · RGB to Grayscale

## Source

LeetGPU: [RGB to Grayscale](https://leetgpu.com/challenges/rgb-to-grayscale)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[04 · Interleave Arrays](../interleave_arrays/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

Three-channel data provides a concrete layout problem that is not equivalent to adding another vector.

## Primary lesson

**Interleaved channel access**

## Primary concepts

- Data layouts
- Memory coalescing
- Vectorized access

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Planned locations, created only when real work starts:

```text
rgb_to_grayscale/
    README.md
    cuda/
    triton/
    pytorch/
    tests/
    benchmarks/
    figures/
    docs/
    results/
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. One pixel per thread using the source coefficients.
2. Compare pixel tiles and aligned channel groups.
3. Inspect compiler treatment of three-channel strides.
4. Measure width and alignment sensitivity.

## Correctness focus

Channel ordering, coefficient precision, image tails, and output conversion rules.

## Things to measure

- Latency
- Pixels/s
- Transaction efficiency
- Register count

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- How does a three-channel stride affect memory transactions?
- Would converting layout cost more than it saves?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
