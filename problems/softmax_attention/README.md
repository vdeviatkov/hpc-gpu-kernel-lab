# 40 · Softmax Attention

## Source

LeetGPU: [Softmax Attention](https://leetgpu.com/challenges/softmax-attention)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[23 · Matrix Multiplication](../matrix_multiplication_fp32/README.md); [32 · Softmax](../softmax/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

A basic attention workload makes the QK, normalization, and probability-times-V stages visible before masking and multiple heads are added.

## Primary lesson

**Attention pipeline decomposition**

## Primary concepts

- Attention
- GEMM
- Numerical stability
- Memory traffic

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/softmax_attention` and benchmark with
`python -m lab.bench softmax_attention`.

```text
softmax_attention/
    README.md       study plan and status
    api.py          public entry point and draft contract
    cases.py        test and benchmark inputs, bytes and FLOPs
    pytorch/        reference (stub)
    cuda/           kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/         Triton kernel (stub)
    jax/            JAX baseline (stub)
    tests/          test_softmax_attention.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Materialized score/probability baseline with source scaling.
2. Use validated GEMM and softmax components.
3. Measure each stage and the complete operation.
4. Document the quadratic intermediate before considering fusion.

## Correctness focus

Q/K/V shapes, source scale, supported rectangular lengths, and precision matching.

## Things to measure

- End-to-end and stage latency
- Temporary memory
- Sequence-length scaling
- Useful FLOP/s

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Which stage dominates at each sequence length?
- How much of peak memory is the score/probability matrix?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
