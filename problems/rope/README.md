# 38 · Rotary Positional Embedding

## Source

LeetGPU: [Rotary Positional Embedding](https://leetgpu.com/challenges/rotary-positional-embedding)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[04 · Interleave Arrays](../interleave_arrays/README.md); [05 · RGB to Grayscale](../rgb_to_grayscale/README.md); [12 · Weight Dequantization](../weight_dequantization/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

RoPE connects earlier layout and vectorization lessons to transformer Q/K tensors without requiring an attention kernel first.

## Primary lesson

**Positional pair layouts**

## Primary concepts

- Data layouts
- Memory coalescing
- Vectorized access
- Positional encoding

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/rope` and benchmark with
`python -m lab.bench rope`.

```text
rope/
    README.md  study plan and status
    api.py     public entry point and draft contract
    cases.py   test and benchmark inputs, bytes and FLOPs
    pytorch/   reference (stub)
    cuda/      kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/    Triton kernel (stub)
    jax/       JAX baseline (stub)
    tests/     test_rope.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Match source pairing and position convention.
2. Coalesced pairwise rotation.
3. Compare table access and value reuse.
4. Explore fusion only with an explicitly matching producer or cache layout.

## Correctness focus

Position indices, rotary dimension, pair convention, head layout, and low-precision error.

## Things to measure

- Latency
- Useful GB/s
- Table-cache behavior
- Register count

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Are pairs adjacent or split across the feature dimension?
- Is sine/cosine preparation inside or outside the measured contract?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
