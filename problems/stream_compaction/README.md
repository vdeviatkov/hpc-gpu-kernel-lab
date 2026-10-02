# 19 · Stream Compaction

## Source

LeetGPU: [Stream Compaction](https://leetgpu.com/challenges/stream-compaction)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[14 · Count Array Element](../count_array_element/README.md); [17 · Prefix Sum](../prefix_sum/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

Compaction composes predicates, prefixes, and scatter, closely resembling token dispatch and sparse preprocessing.

## Primary lesson

**Stable parallel filtering**

## Primary concepts

- Prefix operations
- Irregular memory access
- Synchronization
- Output ownership

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton are useful for explicit memory ownership and scheduling. JAX is deferred: dynamic sizes or mutation would need a carefully matched functional contract before a fair XLA comparison.

Planned locations, created only when real work starts:

```text
stream_compaction/
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

1. Predicate plus prefix plus scatter pipeline.
2. Reduce temporary predicate traffic.
3. Fuse block-local filtering where safe.
4. Sweep retention fractions and output allocation policy.

## Correctness focus

Zero/all retained, tail blocks, stable order, and explicit output-capacity rules.

## Things to measure

- Latency across the pipeline
- Output count
- Traffic and allocations
- Scatter transaction efficiency

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is source ordering preserved?
- Does an apparently faster atomic approach change semantics?
- Who owns the output length?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
