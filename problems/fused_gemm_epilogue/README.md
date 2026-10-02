# 29 · Fused GEMM + Bias + Activation

## Source

**Portfolio Extension** — independently scoped laboratory workload, not a LeetGPU challenge.

This is an original study plan, not a reproduction of the source statement.
For LeetGPU work, confirm the current signature, layout, dtype, and boundary
contract on the linked page before writing a reference. Any broader lab variant
must be labeled separately.

## Prerequisites

[02 · ReLU](../relu/README.md); [25 · General Matrix Multiplication (GEMM)](../gemm_fp16/README.md)

## Difficulty

- LeetGPU: **Not applicable — Portfolio Extension**.

## Why this problem matters

This extension adds a real inference epilogue rather than another GEMM precision alias. The experiment asks when removing output passes matters.

## Primary lesson

**GEMM epilogue fusion**

## Primary concepts

- Kernel fusion
- GEMM
- Mixed precision
- Recomputation vs traffic

## Planned implementations

- [ ] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Planned locations, created only when real work starts:

```text
fused_gemm_epilogue/
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

1. Define bias plus ReLU over a validated GEMM output.
2. Measure separate GEMM and pointwise passes.
3. Fuse bias and activation into the epilogue.
4. Add GELU or BF16 only as explicit numerical variants.

## Correctness focus

Bias axis, rectangular tails, aliasing, output dtype, and exact activation definition.

## Things to measure

- End-to-end latency
- Intermediate bytes avoided
- FLOP/s for GEMM separately
- Register and occupancy changes

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is epilogue traffic significant relative to matrix compute?
- Does fusion increase registers enough to slow the main GEMM?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
