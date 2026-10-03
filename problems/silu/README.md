# 30 · Sigmoid Linear Unit

## Source

LeetGPU: [Sigmoid Linear Unit](https://leetgpu.com/challenges/sigmoid-linear-unit)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[02 · ReLU](../relu/README.md); [06 · Rainbow Table](../rainbow_table/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

SiLU is retained as the activation prerequisite for gated MLPs, with an explicit study of exponential cost rather than another branching exercise.

## Primary lesson

**Special-function throughput**

## Primary concepts

- Instruction mix
- Numerical precision
- Kernel fusion

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/silu` and benchmark with
`python -m lab.bench silu`.

```text
silu/
    README.md  study plan and status
    api.py     public entry point and draft contract
    cases.py   test and benchmark inputs, bytes and FLOPs
    pytorch/   reference (stub)
    cuda/      kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/    Triton kernel (stub)
    jax/       JAX baseline (stub)
    tests/     test_silu.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Direct stable reference.
2. Single-pass GPU activation.
3. Inspect exponential approximation and instruction lowering.
4. Compare standalone cost with later fusion into a gate.

## Correctness focus

Large positive/negative values, near-zero behavior, tails, and output precision.

## Things to measure

- Latency
- Elements/s
- Special-function instruction mix
- Maximum numerical error

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is the exponential pipeline or memory limiting throughput?
- Which approximation changes are acceptable and disclosed?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
