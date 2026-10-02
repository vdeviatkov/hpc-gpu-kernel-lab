# 26 · FP16 Batched Matrix Multiplication

## Source

LeetGPU: [FP16 Batched Matrix Multiplication](https://leetgpu.com/challenges/fp16-batched-matrix-multiplication)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[24 · Batched Matrix Multiplication](../batched_matmul_fp32/README.md); [25 · General Matrix Multiplication (GEMM)](../gemm_fp16/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

This extends the FP32 batching experiment after the Tensor Core lesson, isolating how precision and tile granularity interact with small matrices.

## Primary lesson

**Mixed-precision batch utilization**

## Primary concepts

- GEMM
- Tensor Cores
- Mixed precision
- Work partitioning

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
batched_matmul_fp16/
    README.md
    cuda/
    triton/
    pytorch/
    jax/
    tests/
    benchmarks/
    results/
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Port the documented batching contract.
2. Reuse validated FP16 tile families.
3. Sweep batch count and matrix aspect ratio.
4. Compare padding waste against kernel specialization.

## Correctness focus

Batch strides, odd M/N/K, source output dtype, and accumulator/output rounding.

## Things to measure

- Latency
- Aggregate FLOP/s
- Tensor Core activity
- Tail efficiency

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is low utilization caused by small tiles or by a small batch?
- Does padding to a supported tile pay for the extra computation?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
