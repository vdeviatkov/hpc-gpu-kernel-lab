# 14 · Count Array Element

## Source

LeetGPU: [Count Array Element](https://leetgpu.com/challenges/count-array-element)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[02 · ReLU](../relu/README.md); [13 · Reduction](../reduction/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

Counting is an exact integer reduction and a bridge from elementwise predicates to histograms and dispatch metadata.

## Primary lesson

**Predicate reduction**

## Primary concepts

- Reductions
- Atomics
- Branching and predication

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Planned locations, created only when real work starts:

```text
count_array_element/
    README.md
    cuda/
    triton/
    pytorch/
    tests/
    benchmarks/
    figures/
    docs/
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Predicate evaluation with a simple global counter.
2. Aggregate counts within blocks.
3. Warp-local predicate aggregation.
4. Compare atomics against hierarchical output reduction.

## Correctness focus

No matches, all matches, integer overflow bounds, and exact output count.

## Things to measure

- Latency
- Items/s
- Atomic transactions
- Contention sensitivity

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Does the match fraction alter atomic contention?
- When does a two-pass count beat one atomic per block?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
