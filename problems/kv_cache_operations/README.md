# 47 · KV-Cache Update and Paged Access

## Source

**Portfolio Extension** — independently scoped laboratory workload, not a LeetGPU challenge.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[04 · Interleave Arrays](../interleave_arrays/README.md); [19 · Stream Compaction](../stream_compaction/README.md); [38 · Rotary Positional Embedding](../rope/README.md); [44 · Grouped Query Attention](../grouped_query_attention/README.md)

## Difficulty

- LeetGPU: **Not applicable — Portfolio Extension**.

## Why this problem matters

A cache extension makes mutation, address ownership, page metadata, and per-token latency explicit before attention is fused with cache reads.

## Primary lesson

**Stateful cache layout**

## Primary concepts

- KV cache
- Irregular memory access
- Data layouts
- Output ownership

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton are useful for explicit memory ownership and scheduling. JAX is deferred: dynamic sizes or mutation would need a carefully matched functional contract before a fair XLA comparison.

Planned locations, created only when real work starts:

```text
kv_cache_operations/
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

1. Define contiguous append and gather contracts.
2. Compare sequence-major and head-major storage.
3. Introduce block tables and page-boundary gathers.
4. Measure metadata preparation separately from device updates.

## Correctness focus

Page boundaries, invalid slots, variable lengths, duplicate updates policy, and current-stream/device semantics.

## Things to measure

- Per-token latency
- Useful and actual bytes
- Page-table/L2 behavior
- Allocation and synchronization overhead

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Who owns each cache slot when writes overlap?
- Does paging improve allocation flexibility but harm locality?
- Which CPU costs belong in end-to-end decode timing?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
