# Design

[← Matrix Copy](../README.md) · **Design** · [Results](results.md) · [Testing](testing.md) ·
[Benchmarking](benchmarking.md) · [Profiling](profiling.md)

## Contract

Copy a row-major N × N float32 matrix: `B[i][j] = A[i][j]`, 1 ≤ N ≤ 4096. The source's
performance case is N = 4096. Verified on 2026-10-23 against the LeetGPU challenge source
(`challenges/easy/31_matrix_copy`). The source checker allows 1e-5 tolerance. Here,
equality is **bit exact**: NaN payloads, signed zeros and subnormals must survive.

The API returns a new contiguous tensor and never modifies its input; the source writes
into a caller's `B`. Lab extensions: rectangular M × N, empty matrices, N > 4096, and
FP16/BF16. Strided (pitched) inputs are rejected rather than handled.

## Variants and hypotheses

| Backend | Mechanism | Question |
|---|---|---|
| `pytorch` | `clone`: one `cudaMemcpyAsync` device-to-device | The ceiling: the driver's own copy |
| `cuda_2d` | 2-D grid, (32, threads/32) blocks, `threadIdx.x` along a row | Does a 2-D launch cost anything over a flat one? |
| `cuda_2d_colmajor` | Same grid with `threadIdx.x` down a column | How much does uncoalesced access cost? |
| `cuda` / `cuda_flat` | One thread per element over `m·n` (the source starter's geometry); aliases | The simplest baseline |
| `cuda_vec` | One 16-byte pack per thread, scalar tail | Do fewer, wider requests matter for a pure copy? |
| `cuda_vec_ilp2` / `cuda_vec_ilp4` | 2 / 4 packs per thread, all loads before any store | Does more data in flight per thread help? |
| `cuda_vec_gridstride` | One wave of resident blocks looping over packs | Do fewer blocks help or hurt? |
| `cuda_stream` | Packs with `__ldcs`/`__stcs` (evict-first) | Do cache hints help above L2 size, and hurt below it? |
| `triton` | Flat 1-D blocks of 1024 elements, 4 warps | Triton's equivalent of `cuda_flat` |
| `triton_2d` | 64 × 64 tiles, 4 warps | The tile 08 · Matrix Transpose starts from |

CUDA defaults to 256 threads per block (128/256/512 supported). Every pack variant needs
both buffers 16-byte aligned and falls back to `cuda_flat` otherwise. The 2-D kernels
loop over grid axis y, which is limited to 65535 blocks.

Two compiler effects change what the variants test (see [Results](results.md#sass)):
nvcc unrolls the grid-stride loop 4×, so `cuda_vec_gridstride` also keeps several packs in
flight; and `__ldcs` drops the read-only (`.CONSTANT`) load path the other pack variants
use, so `cuda_stream` changes two things at once.

### Moving bits, not values

A copy never interprets its data. Every kernel moves unsigned integers of the element's
width (`uint32_t` for FP32, `uint16_t` for FP16 and BF16; Triton views the tensors as
int32/int16). Two consequences:

- One instantiation serves FP16 and BF16, so their kernels are identical, and only element
  width can make the dtypes differ.
- Bit patterns pass through unchanged by construction. A float kernel would also preserve
  them in practice (loads and stores do not canonicalize), but that would be a property
  of the generated code, not of the source.

The alternative, `AT_DISPATCH_FLOATING_TYPES_AND2`, would instantiate three identical
kernels and suggest a dtype dependence that does not exist.

## Work and traffic

A copy reads and writes every element once: **2·N²·e bytes** of logical traffic, no
arithmetic. That is also the minimum DRAM traffic when nothing is cached.

The benchmark reuses the input, and the previous call's output is still referenced while
the next one is allocated, so two output blocks alternate. The footprint between calls is
therefore 3·N²·e. Whether a size is L2-resident depends on which of the two output blocks
L2 retains, so the boundary is a band, not a point:

| | 64 MiB L2 filled by 2·N²·e | by 3·N²·e |
|---|---:|---:|
| FP32 | N ≈ 2896 | N ≈ 2365 |
| FP16 / BF16 | N ≈ 4096 | N ≈ 3344 |

The DRAM peak used for `fraction_of_peak` is **960 GB/s**: the RTX 5080's GDDR7 at
30 Gbit/s per pin on a 256-bit bus. It is a nominal number. The practical ceiling is the
measured `copy_reference`, which every table reports.
