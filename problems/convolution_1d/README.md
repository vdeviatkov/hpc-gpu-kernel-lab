# 09 · 1D Convolution

## Source

LeetGPU: [1D Convolution](https://leetgpu.com/challenges/1d-convolution)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[08 · Matrix Transpose](../matrix_transpose/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

A one-dimensional stencil introduces reuse across neighboring outputs and the cost of loading tile halos.

## Primary lesson

**Halo reuse in shared memory**

## Primary concepts

- Shared memory
- Tiled memory access
- Boundary handling

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
convolution_1d/
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

1. Direct per-output stencil with source boundary rules.
2. Load a shared tile and halo.
3. Vary outputs per thread.
4. Compare short-filter unrolling with general loops.

## Correctness focus

Filter orientation, padding mode, boundary outputs, and filters larger than a tile.

## Things to measure

- Latency
- Outputs/s
- Global bytes per output
- Shared memory and register usage

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Does shared staging repay its barriers for short filters?
- How does halo overhead change with tile and filter size?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
