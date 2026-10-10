# 04 · Interleave Arrays

## Source

LeetGPU: [Interleave Arrays](https://leetgpu.com/challenges/interleave-arrays)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
The verified contract is summarized below; broader lab variants are labeled as
extensions.

## Contract

Verified on 2026-10-10 against the statement in LeetGPU's challenge repository
(`challenges/easy/63_interleave`):

```text
out[2·i] = A[i],  out[2·i + 1] = B[i]        for 0 <= i < N    (out has 2N elements)
```

- Contiguous 1-D FP32 inputs of equal length, 1 ≤ N ≤ 50,000,000, written to a separate
  output. The performance test uses N = 25,000,000 with normally distributed inputs; the
  reference is `output[0::2] = A; output[1::2] = B`.
- Lab extensions: FP16/BF16 and N = 0. The function returns a new tensor and never
  modifies its inputs; `a` and `b` may be the same tensor.

## Performance model

| Quantity | Formula | FP32, N = 25,000,000 |
|---|---|---:|
| Memory traffic | read `2·N·s`, write `2·N·s` | 400 MB |
| Work | none: `2·N` element moves | — |
| DRAM-limited time | `4·N·s / 960 GB/s` | 417 µs |
| Practical target | device copy of the same bytes (2N elements) | ≈ 489 µs |

Interleaving moves the same bytes, with the same 1:1 read/write mix, as copying a 2N-element
tensor, so a device copy is the realistic target. The ≈ 489 µs is scaled from the copy
measured in [Reverse Array](../reverse_array/docs/results.md) (977 µs for twice the bytes)
and will be measured directly. Out of place, even 25M FP16 (200 MB moved) is far larger than
the 64 MiB L2, so every benchmark size from 25M up is DRAM-bound.

The difficulty is the shape mismatch: inputs are two contiguous streams, the output
alternates between them. Mapping threads to inputs makes loads contiguous and stores
stride-2; mapping threads to outputs does the opposite. Output position `2i` depends only
on `i`, not on N, so unlike Reverse Array, 16-byte accesses are possible for any N given
aligned pointers; only the last `N mod 4` (FP32) or `N mod 8` (FP16) elements need scalar
code. The reference writes the output in two strided passes, so expect it to be slower
than a copy.

## Prerequisites

[01 · Vector Addition](../vector_add/README.md); [03 · Reverse Array](../reverse_array/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

Packing two streams introduces layout choices that later recur in gated activations and Q/K/V organization.

## Primary lesson

**Lane-to-output mapping**

## Primary concepts

- Indexing
- Memory coalescing
- Data layouts

## Planned implementations

- [x] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/interleave_arrays` and benchmark with
`python -m lab.bench interleave_arrays`.

```text
interleave_arrays/
    README.md       study plan and status
    api.py          public entry point and verified contract
    cases.py        sizes (every N mod 8) and value distributions
    pytorch/        reference (two strided copies)
    cuda/           kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/         Triton kernel (stub)
    tests/          test_interleave_arrays.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Assign one thread to an input pair.
2. Compare output-indexed mapping.
3. Explore aligned paired stores.
4. Sweep tile sizes and tail handling.

## Correctness focus

Uneven launch tails, output ordering, distinct inputs, and storage offsets.

## Things to measure

- Latency
- Useful bytes/s
- Store sectors per request
- Registers/thread

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Which mapping makes both inputs and outputs efficient?
- Do wider stores help after accounting for alignment?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
