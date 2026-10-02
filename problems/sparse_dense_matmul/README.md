# 28 · Sparse Matrix-Dense Matrix Multiplication

## Source

LeetGPU: [Sparse Matrix-Dense Matrix Multiplication](https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[22 · Sparse Matrix-Vector Multiplication](../sparse_matvec/README.md); [23 · Matrix Multiplication](../matrix_multiplication_fp32/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

SpMM turns sparse vector work into a reuse experiment: each sparse value and index can contribute to multiple output features.

## Primary lesson

**Sparse reuse across output columns**

## Primary concepts

- Sparse operations
- Irregular memory access
- Tiled memory access
- Work partitioning

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Planned locations, created only when real work starts:

```text
sparse_dense_matmul/
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

1. Validate the exact source sparse format.
2. Tile dense output columns.
3. Reuse sparse metadata across features.
4. Balance rows with very different nonzero counts.

## Correctness focus

Empty rows, feature tails, skewed sparsity, and duplicate-index semantics.

## Things to measure

- Latency
- Useful nonzero-based FLOP/s
- Index bytes per output
- L2 locality and occupancy

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- When does wider output recover the cost of sparse indexing?
- Does row bucketing help after accounting for preprocessing?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
