# Design

[← Interleave Arrays](../README.md) · **Design** · [Results](results.md) ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

## Contract

Verified on 2026-10-10 against the statement in LeetGPU's challenge repository
(`challenges/easy/63_interleave`,
[LeetGPU Interleave Arrays](https://leetgpu.com/challenges/interleave-arrays), Easy):

```text
out[2·i] = A[i],  out[2·i + 1] = B[i]        for 0 <= i < N    (out has 2N elements)
```

- Contiguous 1-D FP32 inputs of equal length, 1 ≤ N ≤ 50,000,000, written to a separate
  output. The performance test uses N = 25,000,000 with normally distributed inputs; the
  reference is `output[0::2] = A; output[1::2] = B`.
- Lab extensions: FP16/BF16 and N = 0. The function returns a new tensor and never
  modifies its inputs; `a` and `b` may be the same tensor.
- Results must match the reference bit for bit: interleaving only moves values.

[api.py](../api.py) validates both inputs (contiguous 1-D, same length, dtype and device,
supported dtype, no autograd). The C++ bindings repeat the checks and also check the
output size and that it overlaps neither input.

## The mapping problem

Reads and writes have different shapes. If each thread handles one *input* element, loads
are contiguous across a warp but stores land 2 elements apart. If each thread handles one
*output* element, stores are contiguous but neighbouring lanes read alternately from `a`
and `b`. The variants test which mismatch costs time.

A second rule turns out to matter more: memory moves 32-byte sectors, and a sector
written only partly must be merged with its old contents. If both halves of a sector are
written by the same warp at almost the same time, they meet in L2 and the full sector goes
to DRAM once. If they are written in separate passes, the L2 must read the old sector from
DRAM before each write (a read-modify-write).

Output position `2i` depends only on `i`, not on N, so unlike Reverse Array 16-byte
accesses work for any N given 16-byte aligned inputs; only the last `N mod 4` (FP32) or
`N mod 8` (FP16/BF16) elements need scalar code.

## Performance model

| Quantity | Formula | FP32, N = 25,000,000 |
|---|---|---:|
| Memory traffic | read `2·N·s`, write `2·N·s` | 400 MB |
| Work | none: `2·N` element moves | — |
| DRAM-limited time | `4·N·s / 960 GB/s` | 417 µs |
| Practical target | device copy of both inputs | 488 µs |

Interleaving moves the same bytes, with the same 1:1 read/write mix, as copying both
inputs, so that copy (`copy_reference`, which copies every input) is the realistic target.
Out of place, even 25M FP16 (200 MB moved) is far larger than the 64 MiB L2.

## Implementations

| Backend | Mapping | Question it answers |
|---|---|---|
| `cuda_input` | thread `i` reads `a[i]`, `b[i]`, writes `out[2i]`, `out[2i+1]` | Loads contiguous; what do stride-2 stores cost? |
| `cuda_output` | thread `j` writes `out[j]` from `a[j/2]` or `b[j/2]` | Stores contiguous; what do the alternating loads cost? |
| `cuda_pair_store` | thread `i` writes `{a[i], b[i]}` as one aligned 2-element store | Does a paired store (8 bytes FP32, 4 bytes FP16) fix the stride? |
| `cuda_vec` | thread reads one 16-byte pack of `a` and of `b`, writes two 16-byte output packs | Can every access be 16 bytes wide? |
| `pytorch` | `out[0::2] = a; out[1::2] = b` (the reference) | Library baseline: two strided passes |
| `pytorch_stack` | `torch.stack((a, b), 1).reshape(-1)` | Library baseline: one concatenation kernel |

All CUDA kernels use one element (or one pack) per thread and 256 threads per block.
`cuda_vec` falls back to `cuda_input` when any pointer is not 16-byte aligned (inputs that
start mid-allocation); the remaining `N mod k` elements are written by one extra thread.
The shared helpers `Pack`, `load_pack`, `store_pack` and `pack_aligned` come from
`lab/vector.cuh`.

**Triton is deferred.** It was planned (a strided-store kernel and one using
`tl.interleave`) but skipped by decision; the stub stays, and its tests skip as "not
implemented".

Source: [CUDA kernels](../cuda/kernels.cu), [bindings](../cuda/bindings.cpp),
[PyTorch](../pytorch/implementation.py).
