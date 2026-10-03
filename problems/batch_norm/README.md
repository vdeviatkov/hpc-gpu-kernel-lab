# 35 · Batch Normalization

## Source

LeetGPU: [Batch Normalization](https://leetgpu.com/challenges/batch-normalization)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[08 · Matrix Transpose](../matrix_transpose/README.md); [34 · Layer Normalization](../layer_norm/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

The catalog normalizes features across the batch of an N-by-C tensor. This contrasts with row normalization and exposes a different coalescing problem.

## Primary lesson

**Reduction-axis layout**

## Primary concepts

- Reductions
- Data layouts
- Memory coalescing
- Numerical stability

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/batch_norm` and benchmark with
`python -m lab.bench batch_norm`.

```text
batch_norm/
    README.md  study plan and status
    api.py     public entry point and draft contract
    cases.py   test and benchmark inputs, bytes and FLOPs
    pytorch/   reference (stub)
    cuda/      kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/    Triton kernel (stub)
    jax/       JAX baseline (stub)
    tests/     test_batch_norm.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Reference using the source batch statistics.
2. Tile across samples and channels.
3. Compare layout-aware partial statistics.
4. Fuse affine output only after validating variance.

## Correctness focus

N or C equal to one, variance definition, gamma/beta axis, and no accidental running-statistics inference baseline.

## Things to measure

- Latency
- Read/write transactions
- Partial-statistic traffic
- Occupancy and error

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Can lanes coalesce channel accesses while reducing over samples?
- Would a layout conversion save more than it costs?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
