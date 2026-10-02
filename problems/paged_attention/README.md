# 49 · Paged Attention

## Source

**Portfolio Extension** — independently scoped laboratory workload, not a LeetGPU challenge.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[47 · KV-Cache Update and Paged Access](../kv_cache_operations/README.md); [48 · FlashAttention-Style Online Attention](../flash_attention/README.md)

## Difficulty

- LeetGPU: **Not applicable — Portfolio Extension**.

## Why this problem matters

Paged attention combines online normalization with noncontiguous K/V storage and variable sequence lengths, making it a systems-relevant capstone rather than another mask variant.

## Primary lesson

**Irregular decode-time attention**

## Primary concepts

- Attention
- KV cache
- Irregular memory access
- Online reductions
- Work partitioning

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton are useful for explicit memory ownership and scheduling. JAX is deferred: dynamic sizes or mutation would need a carefully matched functional contract before a fair XLA comparison.

Planned locations, created only when real work starts:

```text
paged_attention/
    README.md
    cuda/
    triton/
    pytorch/
    tests/
    benchmarks/
    figures/
    docs/
    results/
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Explicit gather-then-attend reference.
2. Fuse page-table traversal with attention reads.
3. Split long contexts with a correct merge of partial states.
4. Compare scheduling policies across mixed context lengths.

## Correctness focus

Last-page tails, empty/short contexts, mixed lengths, shared-page read ownership, and GQA head mapping.

## Things to measure

- Decode latency distribution
- K/V and metadata traffic
- Temporary memory
- Load balance across requests

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is page indirection or memory bandwidth dominant?
- Does split-context parallelism repay its merge cost?
- Are cache allocation and host scheduling included or excluded explicitly?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
