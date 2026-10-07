# Design

[← ReLU](../README.md) · **Design** · [Results](results.md) ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

## Contract

Verified on 2026-10-07 against the statement in LeetGPU's challenge repository
(`challenges/easy/21_relu`, [LeetGPU ReLU](https://leetgpu.com/challenges/relu), Easy):

```text
out[i] = max(0, x[i])        for 0 <= i < N
```

- Contiguous 1-D FP32, 1 ≤ N ≤ 100,000,000. The performance test uses N = 25,000,000
  with inputs uniform in [−100, 100]; the reference is `torch.relu`.
- Lab extensions: FP16/BF16 and N = 0.
- Results must match `torch.relu` bit for bit. `torch.relu` keeps −0.0 and NaN, so only
  strictly negative values become 0 and the formula is `x < 0 ? 0 : x`.
  `fmaxf(x, 0)` returns +0.0 for both, which is why `cuda_fmax` is kept only as a
  comparison.

[api.py](../api.py) validates the input (contiguous 1-D tensor, supported dtype) and
routes to a backend. The C++ bindings repeat the checks.

## Performance model

| Quantity | Formula | FP32, N = 25,000,000 |
|---|---|---:|
| Memory traffic | `2·N·s` (one read, one write) | 200 MB |
| Work | `N` compare/selects | 25 M |
| Arithmetic intensity | `1 / (2·s)` operations per byte | 1/8 |
| DRAM-limited time | `2·N·s / 960 GB/s` | 208 µs |
| Practical target | device copy of the same tensor | 244.2 µs |

ReLU moves exactly the bytes of a copy, with the same 1:1 read/write mix, so a device
copy is the realistic speed limit, measured in every benchmark run as `copy_reference`.
The work is negligible: every implementation is memory-bound, and variants can only
differ in how many instructions they need to keep memory busy.

## Implementations

All CUDA kernels process one element, or one 16-byte pack, per thread; vector_add showed
that a grid-stride loop only adds registers.

| Backend | Kernel body | Question it answers |
|---|---|---|
| `cuda_select` | `out = x < 0 ? 0 : x` | Baseline: is a ternary a branch-free select? |
| `cuda_branch` | `if (x < 0) out = 0; else out = x;` | Does an explicit `if` produce a real branch? |
| `cuda_fmax` | `out = fmaxf(x, 0)` | One min/max instruction, but different NaN/−0.0 results |
| `cuda_vec` | select on 16-byte packs | Does load width matter (4 FP32 or 8 FP16 per access)? |
| `triton` | `tl.where(x < 0, 0, x)` on masked tiles | What does a tile compiler generate? |
| `pytorch` | `torch.relu` | Library baseline |

`cuda_vec` falls back to `cuda_select` when either pointer is not 16-byte aligned; the
remaining `N mod 4` (or `mod 8`) elements are handled by one extra thread. Triton uses
1024 elements and 4 warps per program; a sweep of 256–4096 elements and 4/8 warps changed
times by at most 2% except one case (see [results](results.md#triton)).

Source: [CUDA kernels](../cuda/kernels.cu), [bindings](../cuda/bindings.cpp),
[Triton](../triton/implementation.py), [PyTorch](../pytorch/implementation.py).
