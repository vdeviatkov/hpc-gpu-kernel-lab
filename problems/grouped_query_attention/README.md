# 44 · Grouped Query Attention

## Source

LeetGPU: [Grouped Query Attention](https://leetgpu.com/challenges/grouped-query-attention)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[42 · Multi-Head Attention](../multi_head_attention/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

GQA is a flagship inference workload: sharing key/value heads can reduce memory traffic only if the implementation avoids physically expanding them.

## Primary lesson

**Shared K/V head reuse**

## Primary concepts

- Attention
- Data layouts
- Cache behavior
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
grouped_query_attention/
    README.md
    cuda/
    triton/
    pytorch/
    jax/
    tests/
    benchmarks/
    figures/
    docs/
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Reference with explicit query-to-K/V head mapping.
2. Kernel with logical sharing instead of materialized repetition.
3. Tile query heads to reuse K/V where resources allow.
4. Separate decode-like and prefill-like experiments.

## Correctness focus

Head-ratio divisibility, query/key lengths, source mask/scaling, and cache-layout lab variants.

## Things to measure

- Latency
- Actual K/V bytes
- L2 behavior
- Scaling by head-group ratio

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is K/V being reread for each query head?
- Does grouping increase reuse but reduce parallelism?
- Are broadcast copies included in the baseline?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
