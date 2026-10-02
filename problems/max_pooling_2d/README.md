# 11 · 2D Max Pooling

## Source

LeetGPU: [2D Max Pooling](https://leetgpu.com/challenges/2d-max-pooling)

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

Pooling replaces weighted accumulation with a neighborhood maximum, isolating stride-dependent reuse and boundary masks.

## Primary lesson

**Overlapping window locality**

## Primary concepts

- Tiled memory access
- Cache behavior
- Boundary handling

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Planned locations, created only when real work starts:

```text
max_pooling_2d/
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

1. Direct output-window traversal.
2. Tile overlapping input regions.
3. Compare stride equal to window against overlapping windows.
4. Tune outputs per thread.

## Correctness focus

Window/stride contract, edges, tie handling, and exceptional values.

## Things to measure

- Latency
- Output elements/s
- Input traffic per output
- Active lanes

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is there enough overlap to justify shared memory?
- Which window/stride combinations leave most lanes idle?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
