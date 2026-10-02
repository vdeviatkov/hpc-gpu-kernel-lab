# 03 · Reverse Array

## Source

LeetGPU: [Reverse Array](https://leetgpu.com/challenges/reverse-array)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[01 · Vector Addition](../vector_add/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

An in-place permutation forces ownership reasoning: each swap must have one writer pair, even though both addresses are globally visible.

## Primary lesson

**Race-free in-place indexing**

## Primary concepts

- Indexing
- In-place ownership
- Memory coalescing

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Planned locations, created only when real work starts:

```text
reverse_array/
    README.md
    cuda/
    triton/
    pytorch/
    tests/
    benchmarks/
    results/
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Assign one thread to each disjoint pair.
2. Handle the center element explicitly.
3. Compare grid-stride and contiguous pair assignments.
4. Inspect both directions of global access.

## Correctness focus

Odd/even length, a single element, and disjoint ownership of every swap.

## Things to measure

- Latency
- Effective GB/s
- Memory transactions
- Sanitizer findings

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Can two threads update the same location?
- Are descending addresses necessarily uncoalesced?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
