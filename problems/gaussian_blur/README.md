# 10 · Gaussian Blur

## Source

LeetGPU: [Gaussian Blur](https://leetgpu.com/challenges/gaussian-blur). Statement and checker
verified on 2026-11-17 from the public challenge repository
([`28_gaussian_blur`](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/28_gaussian_blur),
commit 7691ca5); see [design §0](docs/design.md#0-source-verification).

- A 2-D, possibly rectangular kernel (KH, KW odd, 3 to 21) is passed, not σ.
- Zero padding, same-size output, cross-correlation.
- FP32, images up to 4096 × 4096, checker `F.conv2d` with `atol = rtol = 1e-5`.
- Performance case: 512 × 512 with a 7 × 7 kernel. That kernel is uniform random, so it is
  **neither normalized nor separable**, despite the statement's "Gaussian" and "sums to 1".

**Lab contract** ([design §1](docs/design.md#1-algorithm-and-contract)): the source
operation, widened to FP16/BF16 and KH, KW ≥ 1, accumulated in FP32 and rounded once. A
backend may use two 1-D passes only when `api.separable_factors` accepts the kernel, with an
FP32 intermediate.

## Prerequisites

[09 · 1D Convolution](../convolution_1d/README.md)

## Difficulty

- LeetGPU: **Medium**.

## Why this problem matters

This is retained as a distinct algorithmic experiment: separability can save arithmetic while introducing an intermediate image and another launch.

## Primary lesson

**Separable filtering tradeoffs**

## Primary concepts

- Tiled memory access
- Shared memory
- Recomputation vs traffic

## Planned implementations

- [x] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices
- [ ] JAX/XLA comparison with compilation separated from execution

JAX is included to examine XLA lowering/fusion or a meaningful high-level numerical baseline. A GPU comparison must use the same device, values, dtype, and completion scope.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/gaussian_blur` and benchmark with
`python -m lab.bench gaussian_blur`.

```text
gaussian_blur/
    README.md   study plan and status
    api.py      public entry point, contract, separability test
    cases.py    test and benchmark inputs, bytes and FLOPs
    pytorch/    F.conv2d reference (TF32 off)
    cuda/       kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/     Triton kernel (stub)
    jax/        JAX baseline (stub)
    tests/      test_gaussian_blur.py
    docs/       design.md
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Direct two-dimensional filter with exact boundary semantics.
2. Shared image tile and halo.
3. Use two separable passes only when the supplied kernel is separable.
4. Compare intermediate traffic against saved arithmetic.

## Correctness focus

See [design §2](docs/design.md#2-correctness-reference).

- **Reference:** cuDNN `F.conv2d` with TF32 off.
- **FP64 oracle:** an fsum oracle checked against the FP32-accumulation bound, which grows
  with KH·KW. Edge cases: 1 × 1 images, kernels larger than the image, borders, odd W,
  misaligned rows.
- **Separability:** detected by the FP32-factor residual of the SVD rank-1 term, which must
  be within 4 u_d of the kernel's norm. Gaussians land below 1 u_d; the source's kernels
  land at 0.2 relative.
- **Factoring stays within tolerance:** the two-pass result is within both the fixed
  tolerances and its own KH + KW bound.
- **Every backend is checked against the kernel-scaled bound.** The fixed FP16/BF16
  tolerances cannot catch reduced-precision accumulation at 7 × 7, so a fixed tolerance
  alone is not enough.

## Things to measure

- Latency
- Pixels/s
- Logical and measured traffic
- Launch count

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Is the supplied filter actually separable within the required tolerance?
- When do two passes lose to a fused direct tile?

## Status

🟨 **In Progress.** Contract, PyTorch reference and correctness suite done; CUDA, Triton and
JAX backends not implemented yet.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
