# 20 · Parallel Merge

## Source

LeetGPU: [Parallel Merge](https://leetgpu.com/challenges/parallel-merge)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[17 · Prefix Sum](../prefix_sum/README.md); [19 · Stream Compaction](../stream_compaction/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

Merging sorted arrays introduces data-dependent partitioning and load balance. This is the live catalog match for the preliminary merge-sorted-arrays topic.

## Primary lesson

**Balanced irregular partitioning**

## Primary concepts

- Irregular memory access
- Divergence
- Work partitioning

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants

CUDA receives the algorithm work; PyTorch supplies an exact/reference path. Triton and JAX are deferred because integer dependency analysis or irregular partitioning is the primary lesson, rather than backend coverage.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/parallel_merge` and benchmark with
`python -m lab.bench parallel_merge`.

```text
parallel_merge/
    README.md    study plan and status
    api.py       public entry point and draft contract
    cases.py     test and benchmark inputs, bytes and FLOPs
    pytorch/     reference (stub)
    cuda/        kernels.cu, bindings.cpp, implementation.py (stubs)
    tests/       test_parallel_merge.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Correct partition-and-merge baseline.
2. Balanced diagonal or merge-path partitions.
3. Tune local merge tiles.
4. Compare duplicates, disjoint ranges, and interleaved values.

## Correctness focus

One empty input, unequal lengths, duplicates, and exact non-decreasing output.

## Things to measure

- Latency
- Merged elements/s
- Active/eligible warps
- Partition overhead

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Can every output interval be assigned without overlap?
- Does partitioning overhead dominate short arrays?
- How is duplicate ordering defined?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
