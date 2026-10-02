# 23 · Matrix Multiplication

## Source

LeetGPU: [Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[08 · Matrix Transpose](../matrix_transpose/README.md); [16 · Dot Product](../dot_product/README.md); [21 · Dense GEMV](../dense_gemv/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

The first dense GEMM isolates reuse and tiling in FP32 before introducing Tensor Core precision and fragment constraints.

## Primary lesson

**Shared-memory data reuse**

## Primary concepts

- Shared memory
- Tiled memory access
- Arithmetic intensity
- GEMM

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
matrix_multiplication_fp32/
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

1. One output element per thread.
2. Shared tiles of both operands.
3. Register tiles with several outputs per thread.
4. Tune tile geometry across rectangular shapes.

## Correctness focus

Rectangular matrices, M/N/K tails, FP32/TF32 baseline policy, and accumulation tolerance.

## Things to measure

- Latency
- Useful FLOP/s
- DRAM traffic
- Registers and shared bytes/block

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- How much reuse does each tile create?
- Does a larger tile reduce traffic but also reduce residency?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
