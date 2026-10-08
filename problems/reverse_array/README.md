# 03 · Reverse Array

## Source

LeetGPU: [Reverse Array](https://leetgpu.com/challenges/reverse-array)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
The verified contract is summarized below; broader lab variants are labeled as
extensions.

## Contract

Verified on 2026-10-08 against the statement in LeetGPU's challenge repository
(`challenges/easy/19_reverse_array`):

```text
x[i] <-> x[N - 1 - i]        for 0 <= i < N / 2, in place
```

- Contiguous 1-D FP32, 1 ≤ N ≤ 100,000,000; the result is stored back into the input.
  The performance test uses N = 25,000,000 with inputs uniform in [−1000, 1000]; the
  reference is `input[:] = torch.flip(input, [0])`.
- For odd N the middle element stays in place.
- Lab extensions: FP16/BF16 and N = 0. The function returns the input tensor itself.
- Race-free rule: each pair `(i, N−1−i)` has exactly one owner, which reads both
  elements before writing either. A thread per element writing `x[i] = x[N−1−i]` would
  overwrite values another thread has not read yet.

## Performance model

| Quantity | Formula | FP32, N = 25,000,000 |
|---|---|---:|
| Memory traffic | `2·N·s` (every element read once, written once) | 200 MB |
| Work | `N / 2` swaps, no arithmetic | 12.5 M |
| DRAM-limited time | `2·N·s / 960 GB/s` | 208 µs |
| Practical target | device copy of the same tensor | ~244 µs |
| PyTorch reference | `torch.flip` + `copy_`: `4·N·s` | ~2× the target |

Reversal moves exactly the bytes of a copy, so a device copy is the realistic target, as
for [ReLU](../relu/docs/design.md#performance-model). The PyTorch reference first writes
a reversed copy, then copies it back, so it moves twice the minimum. The back half is
accessed in descending order: within a warp the addresses still cover the same cache
lines, but the mirrored segment straddles a line boundary unless N lines up, and a
16-byte vector access to the mirrored half is aligned only when N is a multiple of 4
(FP32) or 8 (FP16/BF16). The benchmark therefore includes aligned and misaligned N:
25,000,000/25,000,001 (the source's size) and 100,000,000/100,000,001. In place, 25M
FP16 elements are only 50 MB and stay in the 64 MiB L2 between calls, so FP16 needs the
100M sizes to measure DRAM.

## Prerequisites

[01 · Vector Addition](../vector_add/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

An in-place permutation forces ownership reasoning: each swap must have one writer pair, even though both addresses are globally visible.

## Primary lesson

**Race-free in-place indexing**

## Primary concepts

- Indexing
- In-place ownership
- Memory coalescing

## Planned implementations

- [x] PyTorch correctness reference and meaningful baseline
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants
- [ ] Triton implementation with explicit tile/warp choices

CUDA and Triton provide useful low-level contrasts against PyTorch. JAX is deferred until it would answer a distinct compiler or performance question rather than duplicate coverage.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/reverse_array` and benchmark with
`python -m lab.bench reverse_array`.

```text
reverse_array/
    README.md   study plan and status
    api.py      public entry point and verified contract
    cases.py    sizes (odd/even, boundaries, N mod 8) and value ranges
    pytorch/    reference (torch.flip + copy_)
    cuda/       kernels.cu, bindings.cpp, implementation.py (stubs)
    triton/     Triton kernel (stub)
    tests/      test_reverse_array.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Assign one thread to each disjoint pair.
2. Handle the center element explicitly.
3. Compare grid-stride and contiguous pair assignments.
4. Inspect both directions of global access.

## Correctness focus

Odd/even length, a single element, and disjoint ownership of every swap.

## Things to measure

- Latency
- Effective GB/s
- Memory transactions
- Sanitizer findings

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Can two threads update the same location?
- Are descending addresses necessarily uncoalesced?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
