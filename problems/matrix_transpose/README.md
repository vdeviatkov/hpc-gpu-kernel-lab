# 08 · Matrix Transpose

## Source

LeetGPU: [Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[07 · Matrix Copy](../matrix_copy/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

Transpose is a flagship because one small operation exposes load/store coalescing, shared-memory layout, synchronization, and a useful bandwidth control.

## Primary lesson

**Shared-memory bank conflicts**

## Primary concepts

- Memory coalescing
- Shared memory
- Bank conflicts
- Tiled memory access

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/matrix_transpose` and benchmark with
`python -m lab.bench matrix_transpose`.

```text
matrix_transpose/
    README.md      study plan and status
    api.py         public entry point and draft contract
    cases.py       test and benchmark inputs, bytes and FLOPs
    pytorch/       reference (stub)
    cuda/          kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/        Triton kernel (stub)
    tests/         test_matrix_transpose.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Naive global-memory transpose.
2. Shared-memory tiled transpose.
3. Pad the shared tile and compare bank conflicts.
4. Explore vector loads/stores and block dimensions.

## Correctness focus

Tile tails, rectangular lab variants, padded layouts, and synchronization for inactive lanes.

## Things to measure

- Latency and effective GB/s
- Load/store sectors
- Shared bank conflicts
- Occupancy and shared bytes/block

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Are global loads and stores both coalesced?
- Does padding change measured conflicts and latency?
- How close is performance to the copy control?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
