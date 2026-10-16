# 06 · Rainbow Table

## Source

LeetGPU: [Rainbow Table](https://leetgpu.com/challenges/rainbow-table)

Catalog title and difficulty verified on 2026-09-12.

This is an original study plan, not a reproduction of the source statement.
The verified contract is summarized below; broader lab variants are labeled as
extensions.

## Contract

Verified on 2026-10-16 against the statement in LeetGPU's challenge repository
(`challenges/easy/24_rainbow_table`):

```text
fnv1a(x):  h = 2166136261
           for byte k = 0, 1, 2, 3 of x (least significant first):
               h = (h ^ byte_k) · 16777619        (mod 2^32)
out[i] = fnv1a applied R times to input[i]     (each round hashes the previous result)
```

- int32 input of N values in [0, 2³¹ − 1], uint32 output; 1 ≤ N ≤ 10,000,000,
  1 ≤ R ≤ 100. The performance test is N = 5,000,000, R = 10. Results must match exactly.
- Here: a contiguous 1-D int32 tensor and a Python int `rounds` in, a new uint32 tensor out.
- Lab extensions: N = 0; R = 0 (the input reinterpreted as uint32) and R > 100; negative
  inputs (the hash reads only the 32-bit pattern).
- The source's reference computes in int64 and masks to 32 bits after every multiply; in
  32-bit unsigned arithmetic the wraparound is free and the mask disappears.

## Performance model

| Quantity | Formula | N = 5M, R = 10 (source) | N = 10M, R = 100 (maximum) |
|---|---|---:|---:|
| Memory traffic | `8·N` bytes (4 read, 4 written) | 40 MB | 80 MB |
| Work | `4·R` dependent (xor, multiply) steps per element | 200 M steps | 4 G steps |
| Integer ops | `8·N·R` | 0.4 G | 8 G |
| Bytes per hash round | `8 / R` | 0.8 | 0.08 |

A round is 4 steps, each `h = (h ^ byte) · prime`, and each step needs the previous one: every
element is a serial chain of `8·R` dependent instructions, plus byte extraction. Memory
traffic does not depend on R, but work grows with it. So R moves the kernel from
memory-bound (R = 1: 8 bytes per round) to integer-bound (R = 100). The source's case sits
near the crossover, and its 40 MB fits in the 64 MiB L2. Throughput is reported as hash
rounds per second (`N·R / time`).

Measured baselines on RTX 5080, from a quick check during setup (not the recorded results):
the int64 eager reference takes 15.3 ms at the source's size (about 20 small kernels per
round); `torch.compile` of the int32 version takes 32 µs there and 524 µs at N = 10M,
R = 100 (1.9 T rounds/s).

## Prerequisites

[01 · Vector Addition](../vector_add/README.md); [02 · ReLU](../relu/README.md)

## Difficulty

- LeetGPU: **Easy**.

## Why this problem matters

Iterated hashing provides an integer-heavy contrast to bandwidth-dominated floating-point kernels, relevant to broader systems engineering.

## Primary lesson

**Integer instruction throughput**

## Primary concepts

- Instruction mix
- Dependency chains
- Occupancy and registers

## Planned implementations

- [x] PyTorch correctness reference (the source's int64 algorithm) and a `torch.compile`
  baseline (int32, all rounds fused)
- [ ] CUDA straightforward baseline
- [ ] CUDA named optimization variants

CUDA receives the algorithm work; PyTorch supplies an exact/reference path. Triton and JAX are deferred because integer dependency analysis or irregular partitioning is the primary lesson, rather than backend coverage.

Scaffold: every stub raises `NotImplementedError` until it is implemented.
Run the tests with `python -m pytest problems/rainbow_table` and benchmark with
`python -m lab.bench rainbow_table`.

```text
rainbow_table/
    README.md   study plan and status
    api.py      public entry point and verified contract
    cases.py    lengths (every n mod 8), round counts, value distributions, R sweep
    pytorch/    reference (source algorithm, int64) and torch.compile baseline
    cuda/       kernels.cu, bindings.cpp, implementation.py (stubs)
    tests/      test_rainbow_table.py
```

## Optimization roadmap

These are candidate experiments, not promised improvements or completed code.

1. Match the exact hash and overflow contract.
2. Compare looped and bounded-unrolled rounds.
3. Process multiple independent items per thread.
4. Inspect register growth as round count increases.

## Correctness focus

Unsigned overflow, shift semantics, round-count bounds, and exact integer results.

## Things to measure

- Hashes/s
- Integer instruction mix
- Registers/thread
- Eligible warps

Separate source conformance from broader shape/dtype experiments. Use the
[benchmarking plan](../../docs/benchmarking.md) and choose
[profiler metrics](../../docs/profiling.md) for a specific hypothesis.

## Questions to answer

- Are dependency chains or instruction issue limiting throughput?
- When does unrolling reduce occupancy enough to lose?

## Status

⬜ **Planned**

No accepted implementation or measured results for this curriculum entry.
Results pending hardware benchmark.

Follow the [optimization methodology](../../docs/methodology.md). When work
begins, add evidence and update the [roadmap status](../../README.md#roadmap).
