# 27 · INT8 Quantized MatMul

## Source

LeetGPU: [INT8 Quantized MatMul](https://leetgpu.com/challenges/int8-quantized-matmul)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[12 · Weight Dequantization](../weight_dequantization/README.md); [25 · General Matrix Multiplication (GEMM)](../gemm_fp16/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

Integer matrix math is valuable only when zero points, scaling, rounding, and saturation agree; hardware throughput cannot excuse a changed quantization function.

## Primary lesson

**Quantized arithmetic contracts**

## Primary concepts

- Quantization
- Tensor Cores
- Mixed precision
- GEMM

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/int8_quantized_matmul` and benchmark with
`python -m lab.bench int8_quantized_matmul`.

```text
int8_quantized_matmul/
    README.md           study plan and status
    api.py              public entry point and draft contract
    cases.py            test and benchmark inputs, bytes and FLOPs
    pytorch/            reference (stub)
    cuda/               kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/             Triton kernel (stub)
    tests/              test_int8_quantized_matmul.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Wide-accumulator correctness reference for the source zero-point formula.
2. Straightforward integer GPU mapping.
3. Supported integer dot/Tensor Core instructions.
4. Fuse scaling and saturation with explicit rounding.

## Correctness focus

Signed INT8, nonzero zero points, scales, clipping boundaries, ties, and accumulator range.

## Things to measure

- Latency
- Integer operations/s
- Tensor Core or integer instruction activity
- Exact mismatch and saturation counts

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Are zero-point correction costs material?
- Can the accumulator overflow for the supported range?
- Is the output rounding rule identical across backends?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
