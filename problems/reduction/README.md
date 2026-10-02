# 13 · Reduction

## Source

LeetGPU: [Reduction](https://leetgpu.com/challenges/reduction)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[08 · Matrix Transpose](../matrix_transpose/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

Reduction is the central prerequisite for losses, norms, softmax, GEMV, and attention. It deserves a controlled comparison of communication strategies.

## Primary lesson

**Warp-level reduction**

## Primary concepts

- Reductions
- Shared memory
- Warp primitives
- Synchronization

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
reduction/
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

1. Straightforward hierarchical reduction.
2. Shared-memory tree.
3. Warp shuffles with shared warp totals.
4. Multiple elements or independent accumulators per thread.

## Correctness focus

Empty lab inputs, warp/block tails, large odd lengths, cancellation, and accumulation/output precision.

## Things to measure

- Latency across all passes
- Effective GB/s
- Barrier stalls
- Registers and eligible warps

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- When is a second launch more expensive than saved work?
- Does reducing barriers improve the actual bottleneck?
- How does summation order affect error?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
