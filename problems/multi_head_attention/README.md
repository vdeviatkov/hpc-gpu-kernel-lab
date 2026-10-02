# 42 · Multi-Head Attention

## Source

LeetGPU: [Multi-Head Attention](https://leetgpu.com/challenges/multi-head-attention)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[24 · Batched Matrix Multiplication](../batched_matmul_fp32/README.md); [40 · Softmax Attention](../softmax_attention/README.md); [41 · Causal Self-Attention](../causal_attention/README.md)

## Difficulty

- LeetGPU: **Hard**.

## Why this problem matters

Multi-head self-attention is a flagship because layout, head dimension, and launch organization interact with every earlier GEMM and reduction lesson.

## Primary lesson

**Head layout and scheduling**

## Primary concepts

- Attention
- Data layouts
- Work partitioning
- Mixed precision

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
multi_head_attention/
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

1. Validate source head splitting and output ordering.
2. Map heads and batches into the launch grid.
3. Compare materialized and reusable head layouts.
4. Tune head-dimension-specific kernels without hiding transform costs.

## Correctness focus

Head divisibility, scale, source mask behavior, output layout, and dtype/accumulator precision.

## Things to measure

- Whole-operation latency
- Layout-conversion bytes
- Occupancy
- Head and sequence scaling

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Are transpose/reshape costs really absent or merely outside timing?
- Do small heads underutilize the same tile that works for large heads?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
