# 17 · Prefix Sum

## Source

LeetGPU: [Prefix Sum](https://leetgpu.com/challenges/prefix-sum)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[13 · Reduction](../reduction/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

Scan differs from reduction because every output depends on a prefix, preparing for compaction and routing offsets.

## Primary lesson

**Cross-block prefix propagation**

## Primary concepts

- Prefix operations
- Warp primitives
- Shared memory
- Synchronization

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/prefix_sum` and benchmark with
`python -m lab.bench prefix_sum`.

```text
prefix_sum/
    README.md  study plan and status
    api.py     public entry point and draft contract
    cases.py   test and benchmark inputs, bytes and FLOPs
    pytorch/   reference (stub)
    cuda/      kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/    Triton kernel (stub)
    tests/     test_prefix_sum.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Correct work-efficient block scan.
2. Hierarchical block totals and offset propagation.
3. Warp-level local scans.
4. Multiple values per thread after boundary correctness.

## Correctness focus

Inclusive source semantics, block boundaries, non-power-of-two sizes, and long sequences.

## Things to measure

- Latency including all passes
- Effective GB/s
- Barrier and launch counts
- Temporary memory footprint

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- How are dependencies between blocks resolved without an unsafe global barrier?
- How do inclusive semantics and rounding order affect correctness?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
