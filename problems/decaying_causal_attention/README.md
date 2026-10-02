# 45 · Decaying Causal Attention

## Source

LeetGPU: [Decaying Causal Attention](https://leetgpu.com/challenges/decaying-causal-attention)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[17 · Prefix Sum](../prefix_sum/README.md); [40 · Softmax Attention](../softmax_attention/README.md); [41 · Causal Self-Attention](../causal_attention/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

The source is an unnormalized geometrically decayed causal operation, not softmax attention. It opens a meaningful algorithmic comparison with recurrence-friendly sequence models.

## Primary lesson

**Recurrence vs quadratic materialization**

## Primary concepts

- Attention
- Prefix operations
- Recomputation vs traffic
- Numerical stability

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
decaying_causal_attention/
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

1. Match the unnormalized source expression and scaling.
2. Tiled quadratic baseline.
3. Explore a state-recurrence formulation with an independent reference.
4. Study chunked composition and long-sequence numerical behavior.

## Correctness focus

Decay in the source range, single position, causal diagonal, and no accidental softmax normalization.

## Things to measure

- Latency vs sequence length
- State and temporary memory
- Executed arithmetic
- Error as decay approaches limiting values

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Can a recurrence remove the score matrix without changing the operation?
- How does roundoff accumulate over long contexts?
- Where does chunking recover parallelism?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
