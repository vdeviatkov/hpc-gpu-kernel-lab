# 32 · Softmax

## Source

LeetGPU: [Softmax](https://leetgpu.com/challenges/softmax)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[13 · Reduction](../reduction/README.md); [30 · Sigmoid Linear Unit](../silu/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

Softmax joins maximum and sum reductions with exponentials. It is a flagship prerequisite for attention and stable classification losses.

## Primary lesson

**Numerically stable fused reduction**

## Primary concepts

- Reductions
- Warp primitives
- Kernel fusion
- Numerical stability
- Occupancy and registers

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
softmax/
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

1. Stable multi-pass maximum, exponential, and sum reference.
2. Cooperative GPU reductions.
3. Fuse a bounded vector/row while retaining intermediates.
4. Study wide-row tiling and register-spill thresholds.

## Correctness focus

Source vector semantics; batched rows are a labeled lab variant. Extreme logits, non-power-of-two widths, infinities, and NaNs need explicit policies.

## Things to measure

- Latency
- Minimum versus actual bytes
- Register spills
- Exponential throughput and normalization error

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Can intermediates stay in registers without losing residency?
- Is width limited by memory, reductions, or exponentials?
- How do power-of-two padding cliffs affect performance?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
