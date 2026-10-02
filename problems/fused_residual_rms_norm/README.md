# 36 · Fused Residual Add and RMS Norm

## Source

LeetGPU: [Fused Residual Add and RMS Norm](https://leetgpu.com/challenges/fused-residual-add-and-rms-norm)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[01 · Vector Addition](../vector_add/README.md); [33 · RMS Normalization](../rms_norm/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

The source already provides the transformer-relevant weighted row operation. Keeping it as LeetGPU avoids inventing a duplicate portfolio extension.

## Primary lesson

**Residual-normalization fusion**

## Primary concepts

- Kernel fusion
- Reductions
- Recomputation vs traffic
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
fused_residual_rms_norm/
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

1. Separate residual addition and weighted RMSNorm baseline.
2. Fuse addition into statistics accumulation.
3. Compare retaining the sum with recomputing it.
4. Sweep hidden widths and token counts.

## Correctness focus

Per-feature weights, row epsilon, residual aliasing policy, FP32 source semantics, and separately labeled low-precision variants.

## Things to measure

- End-to-end latency
- Intermediate bytes avoided
- Registers and local-memory traffic
- Numerical error

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Does recomputing the sum cost less than retaining it?
- Does the contract require an additional residual output or only the normalized output?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
