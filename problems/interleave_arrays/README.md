# 04 · Interleave Arrays

## Source

LeetGPU: [Interleave Arrays](https://leetgpu.com/challenges/interleave-arrays)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[01 · Vector Addition](../vector_add/README.md); [03 · Reverse Array](../reverse_array/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

Packing two streams introduces layout choices that later recur in gated activations and Q/K/V organization.

## Primary lesson

**Lane-to-output mapping**

## Primary concepts

- Indexing
- Memory coalescing
- Data layouts

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Planned locations, created only when real work starts:

```text
interleave_arrays/
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

1. Assign one thread to an input pair.
2. Compare output-indexed mapping.
3. Explore aligned paired stores.
4. Sweep tile sizes and tail handling.

## Correctness focus

Uneven launch tails, output ordering, distinct inputs, and storage offsets.

## Things to measure

- Latency
- Useful bytes/s
- Store sectors per request
- Registers/thread

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Which mapping makes both inputs and outputs efficient?
- Do wider stores help after accounting for alignment?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
