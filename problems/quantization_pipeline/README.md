# 46 · Quantize / Dequantize Pipeline

## Source

**Portfolio Extension** — independently scoped laboratory workload, not a LeetGPU challenge.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[12 · Weight Dequantization](../weight_dequantization/README.md); [13 · Reduction](../reduction/README.md); [27 · INT8 Quantized MatMul](../int8_quantized_matmul/README.md)

## Difficulty

- LeetGPU: **Not applicable — Portfolio Extension**.

## Why this problem matters

This extension computes scales and quantizes values, beyond the catalog dequantization task that receives scales. It connects throughput to reconstruction quality.

## Primary lesson

**Quantization error vs bandwidth**

## Primary concepts

- Quantization
- Reductions
- Mixed precision
- Kernel fusion

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/quantization_pipeline` and benchmark with
`python -m lab.bench quantization_pipeline`.

```text
quantization_pipeline/
    README.md           study plan and status
    api.py              public entry point and draft contract
    cases.py            test and benchmark inputs, bytes and FLOPs
    pytorch/            reference (stub)
    cuda/               kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/             Triton kernel (stub)
    tests/              test_quantization_pipeline.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Define symmetric per-row INT8 scaling, rounding, and clipping.
2. Separate scale reduction and conversion baseline.
3. Fuse bounded groups where profitable.
4. Compare standalone reconstruction with consumption by matrix kernels.

## Correctness focus

Zero rows, rounding ties, saturation, outliers, nonfinite policy, and group tails. FP8 is future hardware-gated scope, not an implied implementation.

## Things to measure

- Latency and effective GB/s
- Scale overhead
- Reconstruction error
- Saturation counts

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- How do group size and outliers change both error and throughput?
- Does fusion save traffic after accounting for the scale reduction?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
