# 22 · Sparse Matrix-Vector Multiplication

## Source

LeetGPU: [Sparse Matrix-Vector Multiplication](https://leetgpu.com/challenges/sparse-matrix-vector-multiplication)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[21 · Dense GEMV](../dense_gemv/README.md); [18 · Histogramming](../histogramming/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

Sparse GEMV teaches index traffic, row imbalance, and gather locality that dense bandwidth models miss.

## Primary lesson

**Irregular sparse memory access**

## Primary concepts

- Irregular memory access
- Sparse operations
- Work partitioning
- Reductions

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Planned locations, created only when real work starts:

```text
sparse_matvec/
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

1. Match the source sparse representation before optimizing.
2. Compare thread-per-row and warp-per-row mappings where applicable.
3. Bucket work by row length.
4. Inspect index and vector traffic separately.

## Correctness focus

Empty rows, repeated indices if allowed, extreme row lengths, and exact sparse-format contract.

## Things to measure

- Latency
- Nonzeros/s
- Bytes per nonzero
- Row imbalance and L2 behavior

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Do a few long rows dominate completion time?
- Does reordering improve locality after including preprocessing cost?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
