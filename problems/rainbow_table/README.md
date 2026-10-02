# 06 · Rainbow Table

## Source

LeetGPU: [Rainbow Table](https://leetgpu.com/challenges/rainbow-table)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[01 · Vector Addition](../vector_add/README.md); [02 · ReLU](../relu/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

Iterated hashing provides an integer-heavy contrast to bandwidth-dominated floating-point kernels, relevant to broader systems engineering.

## Primary lesson

**Integer instruction throughput**

## Primary concepts

- Instruction mix
- Dependency chains
- Occupancy and registers

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants

CUDA receives the algorithm work; PyTorch supplies an exact/reference path. Triton and JAX are deferred because integer dependency analysis or irregular partitioning is the primary lesson, rather than backend coverage.

Planned locations, created only when real work starts:

```text
rainbow_table/
    README.md
    cuda/
    pytorch/
    tests/
    benchmarks/
    results/
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Match the exact hash and overflow contract.
2. Compare looped and bounded-unrolled rounds.
3. Process multiple independent items per thread.
4. Inspect register growth as round count increases.

## Correctness focus

Unsigned overflow, shift semantics, round-count bounds, and exact integer results.

## Things to measure

- Hashes/s
- Integer instruction mix
- Registers/thread
- Eligible warps

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Are dependency chains or instruction issue limiting throughput?
- When does unrolling reduce occupancy enough to lose?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
