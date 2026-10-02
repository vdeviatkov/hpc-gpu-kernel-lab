# 31 · Swish-Gated Linear Unit

## Source

LeetGPU: [Swish-Gated Linear Unit](https://leetgpu.com/challenges/swish-gated-linear-unit)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[04 · Interleave Arrays](../interleave_arrays/README.md); [30 · Sigmoid Linear Unit](../silu/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

The source splits one even-length vector into two halves. This provides a focused fusion and layout experiment before the full matrix-based SwiGLU MLP.

## Primary lesson

**Elementwise gating fusion**

## Primary concepts

- Kernel fusion
- Data layouts
- Instruction mix

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
swiglu_activation/
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

1. Separate SiLU and multiply baseline on the source halves.
2. Fuse gating into one pass.
3. Compare aligned paired reads and output tiles.
4. Evaluate register use and tails.

## Correctness focus

Even input length, half-length output, half ordering, extreme activations, and FP32 source semantics.

## Things to measure

- Latency
- Intermediate traffic
- Launch count
- Register count

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Does the split-half layout preserve coalescing for both inputs?
- How much of the gain is launch removal versus traffic removal?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
