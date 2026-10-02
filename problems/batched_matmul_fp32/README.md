# 24 · Batched Matrix Multiplication — FP32

## Source

LeetGPU: [Batched Matrix Multiplication](https://leetgpu.com/challenges/batched-matrix-multiplication)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[23 · Matrix Multiplication](../matrix_multiplication_fp32/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

FP32 batching is a flagship because small matrices can be scheduling-bound even when a single large GEMM is compute-bound.

## Primary lesson

**Batch scheduling and occupancy**

## Primary concepts

- GEMM
- Work partitioning
- Occupancy and registers
- Launch overhead

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
batched_matmul_fp32/
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

1. Independent GEMMs with batch indexing.
2. One launch over batch and output tiles.
3. Tune tiles jointly with batch count.
4. Study persistent scheduling only after measuring small-GEMM gaps.

## Correctness focus

Batch one, large batch, rectangular tails, and FP32/TF32 precision controls.

## Things to measure

- Latency
- Aggregate FLOP/s
- SM utilization
- Wave count and launch overhead

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is the bottleneck arithmetic or too few active tiles?
- Does a configuration for one large matrix fail on many small matrices?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
