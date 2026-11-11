# 09 · 1D Convolution

## Source

LeetGPU: [1D Convolution](https://leetgpu.com/challenges/1d-convolution)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[08 · Matrix Transpose](../matrix_transpose/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

A one-dimensional stencil introduces reuse across neighboring outputs and the cost of loading tile halos.

## Primary lesson

**Halo reuse in shared memory**

## Primary concepts

- Shared memory
- Tiled memory access
- Boundary handling

## Planned implementations

- [x] PyTorch correctness reference and meaningful baseline (`F.conv1d`, TF32 off;
  validated against an FP64 oracle; see [design](docs/design.md))
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [x] Triton implementation with explicit tile/warp choices (direct stencil, 8 bytes of
  outputs per thread by default; see [design §4](docs/design.md#4-triton-the-naive-stencil-measured))
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/convolution_1d` and benchmark with
`python -m lab.bench convolution_1d`.

```text
convolution_1d/
    README.md    study plan and status
    api.py       validated public contract
    cases.py     test and benchmark inputs, bytes and FLOPs
    pytorch/     F.conv1d reference (TF32 off)
    cuda/        kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/      direct-stencil Triton kernel
    jax/         JAX baseline (stub)
    tests/       test_convolution_1d.py
    docs/        design, performance model and measurements
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Direct per-output stencil with source boundary rules.
2. Load a shared tile and halo.
3. Vary outputs per thread.
4. Compare short-filter unrolling with general loops.

## Correctness focus

Filter orientation, padding mode, boundary outputs, and filters larger than a tile.

## Things to measure

- Latency
- Outputs/s
- Global bytes per output
- Shared memory and register usage

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Does shared staging repay its barriers for short filters?
- How does halo overhead change with tile and filter size?

## Status

🟨 **In progress**

Contract, PyTorch reference, FP64-oracle tests, performance model and the Triton backend
are done; see [design](docs/design.md). On the RTX 5080: 114 tests passed, and Compute
Sanitizer (memcheck, racecheck, synccheck, initcheck over the Triton tests) reports 0
errors. The LeetGPU source contract is not yet verified. CUDA and JAX backends are not
implemented.

Model in brief: arithmetic intensity is about K/s FLOP/B, and the measured RTX 5080 ridge
is 64 FLOP/B (73 at achieved ceilings). FP32 K = 31 is DRAM-bound and K = 255 sits on the
ridge.

Measured so far:

- **The naive stencil is LSU/L1-bound, not DRAM-bound.** Triton saturates L1 (99.7%)
  while DRAM is 5% busy at K = 255. It runs at 5.0 (FP32) to 6.0 (FP16/BF16) TFLOP/s:
  2.53 and 2.12 ms at N = 25M, K = 255, about 10× the 249 / 218 µs roofline floor. Even at
  K = 31 it is 1.3× (FP32) to 2.3× (FP16) off the DRAM floor.
- **L1 has two limits:** about 0.5 load requests and 1 wavefront per SM per clock. The
  default tile of 8 bytes of outputs per thread balances them, 1.4–1.65× faster than
  1024 outputs × 4 warps.
- **cuDNN is a weak baseline here:** 1.6–9.1× slower than Triton, and 404× slower for FP16
  K = 255, where it falls back to a generic kernel (861 ms).
- Next: register-blocked CUDA variants to cut L1 requests per FMA.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
