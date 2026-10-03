# 02 · ReLU

## Source

LeetGPU: [ReLU](https://leetgpu.com/challenges/relu)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[01 · Vector Addition](../vector_add/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

A simple activation makes it possible to inspect branch lowering without hiding behavior inside a larger fused operator.

## Primary lesson

**Predication and branching**

## Primary concepts

- Indexing
- Branching and predication
- Instruction mix

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/relu` and benchmark with
`python -m lab.bench relu`.

```text
relu/
    README.md  study plan and status
    api.py     public entry point and draft contract
    cases.py   test and benchmark inputs, bytes and FLOPs
    pytorch/   reference (stub)
    cuda/      kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/    Triton kernel (stub)
    tests/     test_relu.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Straightforward elementwise mapping.
2. Compare conditional and max-expression lowering.
3. Vary the sign distribution.
4. Inspect instructions before adding vectorization.

## Correctness focus

Zero, signed zero, tails, and an explicit policy for exceptional floating-point values.

## Things to measure

- Latency
- Elements/s
- Branch and predicate behavior
- Instruction count

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Does the compiler generate a branch or predicated instructions?
- Does input sign distribution change execution?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
