# 33 · RMS Normalization

## Source

LeetGPU: [RMS Normalization](https://leetgpu.com/challenges/rms-normalization)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[13 · Reduction](../reduction/README.md); [32 · Softmax](../softmax/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

RMSNorm is a flagship with a compact reduction-plus-scale structure. The source is one-dimensional with scalar gamma and beta; transformer-style weighted rows are a documented later variant.

## Primary lesson

**Normalization traffic and precision**

## Primary concepts

- Reductions
- Kernel fusion
- Numerical stability
- Mixed precision

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
rms_norm/
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

1. Match source scalar affine and epsilon semantics.
2. Reduce squared values in FP32.
3. Compare rereading input with retaining values.
4. Add clearly separated row/feature-weight and low-precision lab cases.

## Correctness focus

Zero/constant vectors, extreme magnitudes, scalar scale/shift, epsilon, and accumulation precision.

## Things to measure

- Latency
- Effective GB/s
- Register footprint
- Error over input scale

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Where does epsilon belong in the exact contract?
- Does retaining inputs reduce memory traffic at an acceptable register cost?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
