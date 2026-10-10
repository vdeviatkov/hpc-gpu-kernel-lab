# Design

[← Reverse Array](../README.md) · **Design** · [Results](results.md) ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

## Contract

Verified on 2026-10-08 against the statement in LeetGPU's challenge repository
(`challenges/easy/19_reverse_array`,
[LeetGPU Reverse Array](https://leetgpu.com/challenges/reverse-array), Easy):

```text
x[i] <-> x[N - 1 - i]        for 0 <= i < N / 2, in place
```

- Contiguous 1-D FP32, 1 ≤ N ≤ 100,000,000; the result is stored back into the input.
  The performance test uses N = 25,000,000 with inputs uniform in [−1000, 1000]; the
  reference is `input[:] = torch.flip(input, [0])`.
- For odd N the middle element stays in place.
- Lab extensions: FP16/BF16 and N = 0. The function returns the input tensor itself.
- Results must match the reference bit for bit: reversal only moves values.

## Ownership: why in place needs care

If thread `i` simply wrote `x[i] = x[N−1−i]`, it could overwrite a value another thread
has not read yet. The rule every kernel follows: **each pair `(i, N−1−i)` has exactly one
owner, which reads both elements before writing either.** In `cuda_pair` the owner is one
thread (`cuda::std::swap(x[i], x[n−1−i])`). In the tile kernels the owner is a block, and
`__syncthreads()` separates its loads from its stores. In Triton the compiler decides
which thread handles which tile element, so both Triton kernels put a block-wide barrier
(`tl.debug_barrier()`) between loads and stores.

## Alignment: when 16-byte accesses are possible

A 16-byte pack holds `k` elements (4 FP32, 8 FP16/BF16). Front pack `i` starts at element
`i·k`; its mirror starts at `N − k − i·k`. The two starts always sum to `N − k`, so if the
front is pack-aligned, the mirror is aligned **only when N is a multiple of k**. Peeling
elements from the front cannot change that. Consequences:

- `cuda_vec` (pack pairs) needs `N % k == 0` and a 16-byte aligned pointer, otherwise it
  runs `cuda_pair`;
- `cuda_tile` works for any N: only the two ends of each range are misaligned, and they
  are moved element by element.

## Performance model

| Quantity | Formula | FP32, N = 100,000,000 |
|---|---|---:|
| Memory traffic | `2·N·s` (every element read once, written once) | 800 MB |
| Work | `N / 2` swaps, no arithmetic | 50 M |
| DRAM-limited time | `2·N·s / 960 GB/s` | 833 µs |
| Practical target | device copy of the same tensor | 977 µs |
| PyTorch reference | `torch.flip` + `copy_`: `4·N·s` | ~2× the target |

Reversal moves exactly the bytes of a copy, so a device copy is the realistic target, as
for [ReLU](../../relu/docs/design.md#performance-model). In place, 25M FP16 elements are
only 50 MB and stay in the 64 MiB L2 between calls, so FP16 needs N = 100M to measure DRAM.

## Implementations

| Backend | Idea | Question it answers |
|---|---|---|
| `cuda_pair` | one thread per pair, `swap(x[i], x[n−1−i])` | Is descending access still coalesced? |
| `cuda_vec` | one thread per pair of 16-byte packs, reversed within the pack | What do 16-byte accesses gain, when N allows them? |
| `cuda_tile` | each block stages its front range and mirror through shared memory; 16-byte accesses for interior packs, scalar for range ends | Can wide accesses work for any N? |
| `cuda_tile_async` | `cuda_tile` with `cp.async` global→shared copies for interior packs | Does an asynchronous copy help a kernel with nothing to overlap? |
| `triton_pair` | each tile element owns a pair; back half addressed in descending order | What does Triton generate for descending addresses? |
| `triton_flip` | both halves loaded ascending, reversed in registers with `tl.flip` | Can Triton vectorize both halves? |
| `pytorch` | `x.copy_(torch.flip(x, [0]))` | Library baseline |

Launch details: 256 threads per block for the CUDA kernels; the tile kernels use
`2 · threads · 16` bytes of dynamic shared memory (plus 32 bytes for `tile_async`, whose
buffers keep the global alignment so `cp.async` destinations are 16-byte aligned). Triton
uses 1024 elements and 4 warps per program. The shared helpers `Pack`, `load_pack`,
`store_pack` and `pack_aligned` come from `lab/vector.cuh`.

Source: [CUDA kernels](../cuda/kernels.cu), [bindings](../cuda/bindings.cpp),
[Triton](../triton/implementation.py), [PyTorch](../pytorch/implementation.py).
