# 41 · Causal Self-Attention

## Source

LeetGPU: [Causal Self-Attention](https://leetgpu.com/challenges/causal-self-attention)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[40 · Softmax Attention](../softmax_attention/README.md)

## Difficulty

- LeetGPU: **Hard**.

## Why this problem matters

Causality changes both correctness and useful work. It must be understood before an online or paged attention implementation.

## Primary lesson

**Causal masking and triangular work**

## Primary concepts

- Attention
- Masking
- Tiled memory access
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
causal_attention/
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

1. Explicit masked reference.
2. Fuse the mask into score normalization.
3. Skip fully masked tiles with safe boundaries.
4. Compare useful versus executed arithmetic.

## Correctness focus

One token, tile-boundary lengths, diagonal inclusion, and an explicit all-masked-row policy for lab extensions.

## Things to measure

- Latency
- Masked work avoided
- Sequence-length scaling
- Numerical error

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Are diagonal and partial causal tiles correct?
- Does a reported FLOP/s count executed masked work or useful work?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
