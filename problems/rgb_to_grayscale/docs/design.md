# Design

[← RGB to Grayscale](../README.md) · **Design** · [Results](results.md) ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

## Contract

Verified on 2026-10-12 against the statement in LeetGPU's challenge repository
(`challenges/easy/66_rgb_to_grayscale`,
[LeetGPU RGB to Grayscale](https://leetgpu.com/challenges/rgb-to-grayscale), Easy):

```text
gray[p] = 0.299·R[p] + 0.587·G[p] + 0.114·B[p]      for every pixel p of an H × W image
```

- FP32 input of `H·W·3` interleaved values (R, G, B, R, G, B, …), FP32 output of `H·W`
  values; 1 ≤ W, H ≤ 4096, at most 4,194,304 pixels, values in [0, 255]. The exact
  coefficients are required. The performance test is 2048 × 2048 with integers 0–255; the
  tolerance is 1e−5.
- Here: a contiguous `(H, W, 3)` tensor in, a new `(H, W)` tensor out (the same memory
  layout as the source's flat arrays).
- Lab extensions: FP16/BF16 (computed in FP32, rounded to the image dtype), empty images,
  and images above 4,194,304 pixels for DRAM-bound measurements.
- Results are compared within tolerances, not bit for bit: fused multiply-add and summation
  order can change the last bit (FP32 1e−5; FP16 and BF16 one unit in the last place at 255).

[api.py](../api.py) validates the image (tensor, `(H, W, 3)`, contiguous, supported dtype,
no autograd). The C++ bindings repeat the checks and also check the output's shape, dtype
and device.

## The layout problem

Each pixel is three consecutive values: 12 bytes in FP32, 6 in FP16. With one pixel per
thread, neighbouring lanes are 12 bytes apart, so each of the three channel loads is a
stride-3 access: one warp-wide load touches 384 bytes (12 sectors) of which it uses 128.
The question is whether that costs anything, since the next two channel loads use the rest
of the same sectors. Three ways around it:

- **Wider accesses:** 4 FP32 pixels (or 8 FP16) are exactly 48 bytes, three 16-byte packs,
  and their 4 (8) results are one 16-byte pack. A thread can read its pixels with three
  `LDG.E.128` and write them with one `STG.E.128`.
- **Staging in shared memory:** a block copies its tile of the image with contiguous
  16-byte loads, then each thread reads its pixel from shared memory. A stride of 3 words
  is free of bank conflicts in FP32 (3 and 32 share no factor); in FP16 two lanes share a
  word and 32 lanes span 48 words, so some banks serve two words (2-way conflicts).
- **Changing the layout:** convert to planes (all R, then all G, then all B), after which
  every load is contiguous. The input arrives interleaved, so the conversion is part of
  the cost.

## Performance model

| Quantity | Formula | 2048² FP32 (source) | 4096² FP32 (extension) |
|---|---|---:|---:|
| Read | `3·P·s` (P = H·W pixels, s = element size) | 50.3 MB | 201 MB |
| Write | `P·s` | 16.8 MB | 67 MB |
| Work | 3 multiplies + 2 adds per pixel | 21 M ops | 84 M ops |
| DRAM-limited time | `4·P·s / 960 GB/s` | 70 µs | 280 µs |

Five operations per 16 bytes moved: memory-bound like the previous problems. The source's
size, 2048 × 2048, is 67 MB in FP32, about the size of the 64 MiB L2: in FP32 it still
streams from DRAM, while the 34 MB FP16 image stays in L2 between calls. Since 2048 × 2048
is the contract maximum, DRAM behaviour in every dtype needs the lab extension
(4096 × 4096, 268 MB in FP32).

The copy reference (a device copy of the image) reads and writes in a 1:1 mix, while this
kernel reads three bytes per byte written; it moves more bytes than the kernel, so it is a
bandwidth reference rather than a time to reach. Compare GB/s, not percent of copy.

## Implementations

| Backend | Approach | Question it answers |
|---|---|---|
| `cuda_pixel` | one thread per pixel on a 2-D grid (32 × 8 blocks), three scalar loads | What does the stride-3 access cost? |
| `cuda_vec` | one thread per 4 (FP32) or 8 (FP16/BF16) pixels: three 16-byte loads, one 16-byte store | Do wide accesses help? |
| `cuda_smem` | a 256-pixel tile staged through shared memory with 16-byte loads | Does making global loads perfectly contiguous help? |
| `cuda_planar` | `deinterleave_rgb` writes R, G, B planes, then a planar kernel | Would a planar layout help, and is converting worth it? |
| `pytorch` | the formula in eager mode (the reference) | Correctness oracle; library baseline |
| `pytorch_matmul` | `image @ w`: one cuBLAS matrix-vector product | Does one library call beat several? |
| `pytorch_compile` | `torch.compile` of the formula: one fused Triton kernel | What does PyTorch's compiler reach? |

- **`cuda_vec`** runs on a 1-D grid over the whole image: the image is contiguous, so pixel
  group `v` is input pack group `v` and output pack `v` for any width. The thread just past
  the last full group handles the remaining `P mod 4` (`P mod 8`) pixels. The structs
  `Pixel`, `Vector` (`lcm(pixel size, 16)` bytes), `GrayscaleVector`, `loadVector` and
  `storeGrayscaleVector` are defined in the kernel file.
- **`cuda_smem`** copies each full tile with 16-byte packs (3 KB in FP32, 1.5 KB in FP16)
  and the last partial tile element by element, then `__syncthreads()`, then one pixel per
  thread from `tile[3t .. 3t + 2]`.
- `cuda_vec` and `cuda_smem` need a 16-byte aligned image (a fresh tensor always is; a
  row slice of a larger image may not be) and fall back to `cuda_pixel` otherwise.
- **`cuda_planar`** allocates its `3·P` planes from PyTorch's stream-ordered allocator.
  In a profiler the two kernels appear separately, so the second one shows what the
  computation costs on data that is already planar.
- **`pytorch_compile`** compiles one static kernel per image shape and dtype. With
  `dynamic=True` Inductor would generate one kernel for all sizes, but it picks the block
  size from the first image it compiles and keeps that choice in its on-disk cache: after
  a 2 × 2 image, 4096 × 4096 FP32 ran in 1984 µs instead of 312 µs. Static compilation
  raises the recompile limit for each new shape and fails loudly if the limit is reached,
  rather than silently falling back to eager mode. Compilation happens in the warmup and
  is never timed.

All kernels compute in FP32 with float literals (`0.299f`, …): double literals would
promote the expression to FP64, which is off-contract and far slower on this GPU.

**Triton** is not implemented yet: the stub stays, and its tests skip as "not implemented".

Source: [CUDA kernels](../cuda/kernels.cu), [bindings](../cuda/bindings.cpp),
[PyTorch](../pytorch/implementation.py).
