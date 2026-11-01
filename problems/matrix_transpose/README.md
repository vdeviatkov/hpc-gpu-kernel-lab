# 08 · Matrix Transpose

## Source

LeetGPU: [Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)

Source contract verified on 2026-10-24; see [design and performance model](docs/design.md).

This is an original study plan, not a reproduction of the source statement.
Source: row-major fp32, rows and cols from 1 to 8192; performance case 7000×6000.
The lab also supports fp16/bf16, empty matrices and larger dimensions.

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

- [x] PyTorch correctness reference and materialized eager baseline
- [x] CUDA straightforward baselines: `cuda_naive_row`, `cuda_naive_col` (written, not yet run on a GPU)
- [x] CUDA named optimization variants: `cuda_smem`, `cuda_smem_pad`, `cuda_smem_swizzle`,
  `cuda_vec_ilp` (written, not yet run on a GPU)
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Steps 0–3 complete: source verification, algorithm contract, tested correctness
reference, and a simple performance model. Step 4 is written: six CUDA variants, pending
GPU validation. Triton remains a stub.
Run the tests with `python -m pytest problems/matrix_transpose` and benchmark with
`python -m lab.bench matrix_transpose`.

```text
matrix_transpose/
    README.md      study plan and status
    api.py         validated public contract
    cases.py       test and benchmark inputs, bytes and FLOPs
    pytorch/       materialized transpose reference
    cuda/          kernels.cu, bindings.cpp, implementation.py (six variants)
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

🟨 **In Progress**

CPU reference validation: 50 passed; 391 GPU cases skipped (NVIDIA GPU unavailable).
CUDA variants are written but not yet compiled or run on a GPU.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
