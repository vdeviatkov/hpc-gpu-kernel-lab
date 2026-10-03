# 18 · Histogramming

## Source

LeetGPU: [Histogramming](https://leetgpu.com/challenges/histogramming)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[14 · Count Array Element](../count_array_element/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

Histogram performance depends on the input distribution, making it a useful antidote to benchmarks using only uniform random data.

## Primary lesson

**Atomic contention**

## Primary concepts

- Atomics
- Shared memory
- Bank conflicts
- Data-dependent performance

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/histogramming` and benchmark with
`python -m lab.bench histogramming`.

```text
histogramming/
    README.md   study plan and status
    api.py      public entry point and draft contract
    cases.py    test and benchmark inputs, bytes and FLOPs
    pytorch/    reference (stub)
    cuda/       kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/     Triton kernel (stub)
    tests/      test_histogramming.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Global atomics baseline.
2. Block-private shared histograms.
3. Warp aggregation or privatization.
4. Tune bin grouping and partial merge.

## Correctness focus

Uniform, skewed, single-bin, invalid-bin policy, and counter overflow bounds.

## Things to measure

- Latency by distribution
- Atomic transactions
- Shared bank conflicts
- Scratch footprint

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- What happens when all inputs hit one bin?
- Do more private histograms save contention but cost too much shared memory?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
