# 07 · Matrix Copy

## Source

LeetGPU: [Matrix Copy](https://leetgpu.com/challenges/matrix-copy)

Catalog title and difficulty verified on 2026-09-12; contract verified on
2026-10-23 against the LeetGPU challenge source (`challenges/easy/31_matrix_copy`).

**Source contract:** copy an N x N row-major float32 matrix, `B[i][j] = A[i][j]`,
1 ≤ N ≤ 4096; performance is measured at N = 4096. The source checker allows
1e-5 tolerance; here every backend must match bit for bit, NaN payloads and
signed zeros included.

**Lab extensions** (labeled separately): rectangular M x N, empty matrices,
N > 4096, and fp16/bf16 elements. The API returns a new tensor rather than
writing into a caller's `B`, and accepts contiguous inputs only.

## Prerequisites

[01 · Vector Addition](../vector_add/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

A copy is a control experiment for transpose and stencil studies; its bandwidth is a measured local comparison, not a claimed universal hardware peak.

## Primary lesson

**Memory bandwidth ceiling**

## Primary concepts

- Memory coalescing
- Vectorized access
- Cache behavior

## Planned implementations

- [x] PyTorch correctness reference and meaningful baseline (`clone`, a device-to-device memcpy)
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

CUDA and Triton are still stubs that raise `NotImplementedError`; their tests skip.
Run the tests with `python -m pytest problems/matrix_copy` and benchmark with
`python -m lab.bench matrix_copy`.

```text
matrix_copy/
    README.md  study plan and status
    api.py     public entry point and draft contract
    cases.py   test and benchmark inputs, bytes and FLOPs
    pytorch/   reference: clone (cudaMemcpyAsync on the GPU)
    cuda/      kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/    Triton kernel (stub)
    tests/     test_matrix_copy.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Contiguous scalar copy.
2. Aligned vector copy with fallback.
3. Compare flattened and two-dimensional launch geometry.
4. Separate cache-resident and streaming working sets.

## Correctness focus

Square source contract; rectangular, offset, and pitched cases only as labeled lab variants.

## Things to measure

- Latency percentiles
- Effective GB/s
- DRAM bytes
- L2 hit behavior

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is the working set larger than the measured GPU L2?
- How much of transpose overhead is additional to a copy?

## Status

🟨 **In Progress**

Contract, cases, PyTorch reference and tests are done. The tests cover the source
examples and functional inputs, special bit patterns, fresh output, misaligned
input, tails and empty matrices. CUDA and Triton are not implemented.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
