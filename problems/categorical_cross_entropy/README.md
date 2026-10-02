# 37 · Categorical Cross Entropy Loss

## Source

LeetGPU: [Categorical Cross Entropy Loss](https://leetgpu.com/challenges/categorical-cross-entropy-loss)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[15 · Mean Squared Error](../mean_squared_error/README.md); [32 · Softmax](../softmax/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

The live source consumes logits and class indices and averages the loss across the batch. Avoiding full probability materialization makes it an instructive fusion target.

## Primary lesson

**Fused log-sum-exp loss**

## Primary concepts

- Numerical stability
- Reductions
- Kernel fusion
- Irregular memory access

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
categorical_cross_entropy/
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

1. Stable log-sum-exp plus target gather reference.
2. Cooperative per-sample loss.
3. Fuse row statistics and target extraction.
4. Tune the final batch reduction separately.

## Correctness focus

Class-index bounds, one class, extreme logits, batch averaging, and no unstated ignore-index or smoothing behavior.

## Things to measure

- Latency including batch reduction
- Intermediate traffic
- Register footprint
- Loss error on extreme logits

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- How much traffic is saved by avoiding probabilities?
- Does the final batch reduction dominate small classes?
- Is a compared framework loss using identical semantics?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
