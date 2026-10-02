# 43 · Attention with Linear Biases

## Source

LeetGPU: [Attention with Linear Biases](https://leetgpu.com/challenges/attention-with-linear-biases)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[38 · Rotary Positional Embedding](../rope/README.md); [41 · Causal Self-Attention](../causal_attention/README.md); [42 · Multi-Head Attention](../multi_head_attention/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

ALiBi is retained because generating a positional bias inside score tiles can avoid a large materialized bias tensor.

## Primary lesson

**Fused attention bias generation**

## Primary concepts

- Attention
- Positional encoding
- Kernel fusion
- Masking

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
alibi_attention/
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

1. Explicit bias reference with source slope convention.
2. Generate bias from positions inside score tiles.
3. Combine source masking and bias without changing semantics.
4. Compare long-context traffic and instruction cost.

## Correctness focus

Slope construction, head indexing if applicable, causal convention, and long-position numerical range.

## Things to measure

- Latency
- Bias bytes avoided
- Instruction mix
- Error over long positions

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Does on-the-fly bias save bandwidth at the cost of meaningful extra arithmetic?
- Are slope and position signs identical to the source?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
