# 48 · FlashAttention-Style Online Attention

## Source

**Portfolio Extension** — independently scoped laboratory workload, not a LeetGPU challenge.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[25 · General Matrix Multiplication (GEMM)](../gemm_fp16/README.md); [32 · Softmax](../softmax/README.md); [41 · Causal Self-Attention](../causal_attention/README.md); [42 · Multi-Head Attention](../multi_head_attention/README.md); [44 · Grouped Query Attention](../grouped_query_attention/README.md)

## Difficulty

- LeetGPU: **Not applicable — Portfolio Extension**.

## Why this problem matters

This flagship extension asks for exact attention within a documented tolerance while removing the quadratic probability intermediate, with IO reasoning backed by measurements.

## Primary lesson

**Numerically stable online reduction**

## Primary concepts

- Attention
- Online reductions
- Shared memory
- Tensor Cores
- Recomputation vs traffic

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
flash_attention/
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

1. Matched materialized attention baseline.
2. Derive and validate online max/sum state updates.
3. Tile Q/K/V and retain bounded state.
4. Tune resource use and operand staging on a named architecture.

## Correctness focus

Causal and unmasked contracts separated, odd sequence lengths, head tails, dtype error, and an independent reference. Backward is outside initial scope.

## Things to measure

- End-to-end latency
- Peak temporary memory
- Measured DRAM traffic
- Tensor Core activity and register spills

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is online rescaling numerically stable across tiles?
- Which traffic is eliminated rather than cached?
- Where does SRAM/register capacity limit tile size?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
