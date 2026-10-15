# 05 · RGB to Grayscale

## Source

LeetGPU: [RGB to Grayscale](https://leetgpu.com/challenges/rgb-to-grayscale)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
The verified contract is summarized below; broader lab variants are labeled as
extensions.

## Contract

Verified on 2026-10-12 against the statement in LeetGPU's challenge repository
(`challenges/easy/66_rgb_to_grayscale`):

```text
gray[p] = 0.299·R[p] + 0.587·G[p] + 0.114·B[p]      for every pixel p of an H × W image
```

- FP32 input of `H·W·3` interleaved values (R, G, B, R, G, B, …), FP32 output of `H·W`
  values; 1 ≤ W, H ≤ 4096, at most 4,194,304 pixels, values in [0, 255]. The exact
  coefficients are required. The performance test is 2048 × 2048 with integers 0–255;
  the tolerance is 1e−5.
- Here: a contiguous `(H, W, 3)` tensor in, a new `(H, W)` tensor out (the same memory
  layout as the source's flat arrays).
- Lab extensions: FP16/BF16 (computed in FP32, rounded to the image dtype), empty images,
  and images above 4,194,304 pixels for DRAM-bound measurements.
- Results are compared within tolerances, not bit for bit: fused multiply-add and
  summation order can change the last bit.

## Performance model

| Quantity | Formula | 2048 × 2048 FP32 (source) | 4096 × 4096 FP32 (extension) |
|---|---|---:|---:|
| Read | `3·P·s` (P = H·W pixels) | 50.3 MB | 201 MB |
| Write | `P·s` | 16.8 MB | 67 MB |
| Work | 3 multiplies + 2 adds per pixel | 21 M ops | 84 M ops |
| DRAM-limited time | `4·P·s / 960 GB/s` | 70 µs | 280 µs |

Five operations per 16 bytes moved: memory-bound like the previous problems, but now with
arithmetic. The source's size is only 67 MB, about the size of the 64 MiB L2; benchmarks
reuse their input, so much of it stays cached between calls and 2048 × 2048 measures a
warm cache. Since that is already the contract maximum, DRAM behavior needs the lab
extension (4096 × 4096). The copy reference moves bytes in a 1:1 read/write mix while this
kernel reads three bytes per byte written, so the copy is a reference rather than a strict
ceiling, as in Vector Addition.

The layout question: each pixel is 12 bytes, so with one pixel per thread neighbouring
lanes read 12 bytes apart (stride 3). Four FP32 pixels are exactly three 16-byte packs, and
eight FP16 pixels too, so a vectorized kernel can read every pixel with 16-byte loads and
write the results with one 16-byte store. In shared memory a stride of 3 is conflict-free,
because 3 and the 32 banks share no factor.

## Prerequisites

[04 · Interleave Arrays](../interleave_arrays/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

Three-channel data provides a concrete layout problem that is not equivalent to adding another vector.

## Primary lesson

**Interleaved channel access**

## Primary concepts

- Data layouts
- Memory coalescing
- Vectorized access

## Planned implementations

- [x] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/rgb_to_grayscale` and benchmark with
`python -m lab.bench rgb_to_grayscale`.

```text
rgb_to_grayscale/
    README.md      study plan and status
    api.py         public entry point and verified contract
    cases.py       image sizes (every pixel count mod 8) and value distributions
    pytorch/       reference (the source formula in FP32)
    cuda/          kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/        Triton kernel (stub)
    tests/         test_rgb_to_grayscale.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. One pixel per thread using the source coefficients.
2. Compare pixel tiles and aligned channel groups.
3. Inspect compiler treatment of three-channel strides.
4. Measure width and alignment sensitivity.

## Correctness focus

Channel ordering, coefficient precision, image tails, and output conversion rules.

## Things to measure

- Latency
- Pixels/s
- Transaction efficiency
- Register count

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- How does a three-channel stride affect memory transactions?
- Would converting layout cost more than it saves?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
