# 25 · General Matrix Multiplication (GEMM) — FP16

## Source

LeetGPU: [General Matrix Multiplication (GEMM)](https://leetgpu.com/challenges/general-matrix-multiplication-gemm)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[23 · Matrix Multiplication](../matrix_multiplication_fp32/README.md); [24 · Batched Matrix Multiplication](../batched_matmul_fp32/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

The current source GEMM uses FP16 matrices and alpha/beta scaling, satisfying the requested FP16 GEMM topic. It is the main Tensor Core progression, not a duplicate FP32 exercise.

## Primary lesson

**Tensor Core utilization**

## Primary concepts

- GEMM
- Tensor Cores
- Mixed precision
- Shared memory
- Asynchronous copies

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
gemm_fp16/
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

1. Correct SIMT FP16-input baseline with stated accumulator precision.
2. Shared/register tiling.
3. Architecture-supported Tensor Core path.
4. Pipeline operand movement only after profiling stalls.

## Correctness focus

FP16 source input/output, FP32 scalar coefficients, alpha/beta edge cases, K tails, and architecture capability. BF16 is a separately labeled lab variant.

## Things to measure

- Latency and FLOP/s
- Tensor Core activity
- Registers and shared memory
- Numerical error against an explicit reference

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Are dimensions and layouts using Tensor Cores effectively?
- Is operand movement or matrix issue the next bottleneck?
- Are alpha/beta semantics preserved, including reading initial C?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
