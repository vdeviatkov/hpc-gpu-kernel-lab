# Design

[← 1D Convolution](../README.md) · **Design**

## 0. Source verification

**Not verified.** The [LeetGPU challenge](https://leetgpu.com/challenges/1d-convolution)
renders client-side and could not be read on 2026-11-02. Everything below is the lab
contract. Before results are reported as source-conformant, confirm the signature, dtype,
output length, filter orientation and size limits on the live page, then record them
here as 08 · Matrix Transpose does.

## 1. Algorithm and contract

Valid 1-D cross-correlation with no filter flip, which is what frameworks call
convolution:

```text
out[i] = sum_{j=0}^{K-1} x[i + j] * w[j],    0 <= i < N - K + 1
```

- Inputs: contiguous strided 1-D `x` (N) and `w` (K), `1 <= K <= N`, both fp32, fp16 or
  bf16, both on the same device. Storage offsets are allowed, so inputs need not be
  16-byte aligned.
- Output: a fresh contiguous tensor of `N - K + 1` elements, same dtype and device. Inputs
  are not mutated.
- Precision: products are accumulated in FP32 and rounded once to the input dtype. FP16
  and BF16 products are exact in FP32 (22 and 16 significand bits), so for those dtypes the
  only error is FP32 summation and the final rounding.
- Rejected by `api.validate`: other ranks, non-contiguous inputs, mixed dtype or device,
  `K = 0`, `K > N`, fp64/integer/sparse inputs, `requires_grad`, and unresolved
  negative/conjugate views.

## 2. Correctness reference

`F.conv1d(x.view(1, 1, -1), w.view(1, 1, -1)).view(-1)`, run inside
`torch.backends.cudnn.flags(..., allow_tf32=False)` with the other cuDNN flags unchanged.

**The TF32 trap.** PyTorch defaults to `cudnn.allow_tf32 = True`, which lets cuDNN run FP32
convolutions with 10-bit-mantissa products. A TF32 reference would fail correct FP32
kernels by 50–300× the tolerance (measured below). We measured on the RTX 5080 (cuDNN via
PyTorch 2.14.1) at N = 2²⁰ and 25,000,000 and K = 31, 255 and 2047, with and without
`cudnn.benchmark`. cuDNN never chose a TF32 algorithm for this one-channel shape: the error
against FP64 was the same with TF32 allowed and disallowed (≤ 2.2e-6 of max |y|). The
switch stays in the reference anyway, because algorithm choice is not part of cuDNN's
contract and can change with version, shape or `benchmark`.

**FP64 oracle** (`test_reference_matches_oracle`). Explicit loops compute each output as
`math.fsum` of FP64 products. Every input converts to FP64 exactly and a product of two FP32
values is exact in FP64, so each oracle value is the exact sum, correctly rounded. Each
reference output must satisfy the componentwise bound for any FP32-accumulated dot product
rounded once to dtype `d`:

```text
|ŷ − y| ≤ u_d·|y| + (1 + u_d)·γ_K·Σ_j |x[i+j]·w[j]| + η_d,     γ_K = K·u₃₂ / (1 − K·u₃₂)
```

Here `u_d` is the dtype's unit roundoff and `η_d` is half its smallest subnormal. The bound
holds for any summation order, with or without FMA (Higham, *Accuracy and Stability*,
Thm 3.1). On the GPU the test switches TF32 on globally first, to show the reference turns
it off for its own call. Maximum error/bound over `TEST_CASES` on the RTX 5080:

| dtype | K=1 | 3 | 5 | 31 | 255 | 1023 | 2047 (N=2047) | 2047 (N=2100) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 0.003 | 0.167 | 0.055 | 0.093 | 0.005 | 0.001 | 0.000 | 0.001 |
| fp16 | 0.335 | 0.619 | 0.557 | 0.968 | 0.692 | 0.234 | 0.004 | 0.147 |
| bf16 | 0.166 | 0.916 | 0.443 | 0.982 | 0.904 | 0.641 | 0.049 | 0.433 |

FP16 and BF16 approach 1 because a single output rounding can come close to half an ulp. The
bound is rigorous, so this is not flakiness.

`test_reference_contract` checks shape, dtype, device, ownership and non-mutation on offset
inputs with small-integer data. All FP32 partial sums are then exact, so equality with the
exactly rounded sum is exact, and FP16 or BF16 accumulation would fail.
`test_reference_is_correlation_not_convolution` pins the orientation.

### Tolerances

Every backend is compared with the reference under fixed `(rtol, atol)`: fp32 (1e-4, 1e-4),
fp16 (1e-2, 1e-2), bf16 (5e-2, 5e-2). Two correct implementations differ in two ways:

1. **Final rounding.** One FP32 sum may land on the other side of a rounding boundary,
   which is at most 1 ulp of the output: ≤ 2⁻¹⁰ relative in FP16 and 2⁻⁷ in BF16. `rtol`
   absorbs it, independent of K.
2. **FP32 summation order.** The noise grows like √K·u₃₂·√(Σ (x·w)²), which is about
   K·u₃₂ for unit-normal data. This is the only K-dependent term, and it matters only
   near zero, where `atol` has to cover it.

Measured on the RTX 5080 as max |a − b| / (atol + rtol·|b|), where a ratio above 1 fails
(`test_tolerances_cover_summation_order`, `test_tolerances_reject_reduced_precision`).
The order-noise column compares a sequential-FP32 sum with cuDNN, N = K + 4095.

| K | order noise fp32 | fp16 / bf16 | TF32 products (fp32) | FP16 accumulation | BF16 accumulation |
|---:|---:|---:|---:|---:|---:|
| 31 | 0.012 | 0.000 | 50.9 | 1.12 | 1.37 |
| 255 | 0.038 | 0.000 | 108 | 7.4 | 8.3 |
| 2047 | 0.178 | 0.000 | 301 | 38.9 | 67.1 |

The CPU reference gives the same picture: at most 0.24 for FP32 at K = 2047.

**Verdict: the fixed tolerances are enough up to K = 2047, so they are not scaled with K.**
For fp16 and bf16 the K-dependent term is about 10⁻⁴ against an `atol` of 10⁻². It never
decides the result, and the error against FP64 stays flat in K (about 3e-4 FP16, 3e-3 BF16
relative), so it is all output rounding. FP32 has the least headroom: 0.18 at K = 2047, growing
between √K and K (0.012 → 0.038 → 0.178), so filters well past 8K would need
`atol ∝ K·u₃₂·Σ|x·w|`. The
tolerances still reject TF32 at every K and reduced-precision accumulation from K = 31,
narrowly there (1.1–1.4×), comfortably from K = 255.

## 3. Performance model

### Hardware ceilings (measured on cuda.0, 2026-11-02)

| Quantity | Value | Source |
|---|---:|---|
| SMs × FP32 lanes | 84 × 128 | device properties |
| Sustained SM clock under an FFMA loop | 2827 MHz (max 3105) | nvidia-smi sampled during the loop |
| FP32 FFMA peak at that clock | **60.8 TFLOP/s** | 84 · 128 · 2 · 2.827 GHz |
| Achieved FFMA throughput | 58.6 TFLOP/s (96%) | 8 independent FFMA chains/thread, 2 s run |
| DRAM peak at the loaded memory clock | **947 GB/s** | 2 · 14801 MHz · 256 bit / 8 (960 at the 15001 MHz max) |
| Achieved device-to-device copy | 802 GB/s | 1 GiB `copy_`, read + write |
| **Ridge point** | **64.2 FLOP/B** peak/peak, 73.0 achieved/achieved | |

The estimate of 55–60 FLOP/B assumed a lower boost clock. At the clock the card sustains,
the ridge is about 64 FLOP/B, and about 73 for what is achievable in practice.

### Per-output work and intensity

Each output needs `2K` FLOPs and, at minimum, `2s` bytes: one `x` element read and one
output written. `w` is K·s bytes in total and negligible. Arithmetic intensity is therefore

```text
AI = 2K / 2s = K / s   FLOP/B      (FP32: K/4; FP16/BF16 with FP32 FFMA: K/2)
```

`cases.flops` and `cases.logical_bytes` count exactly this: 2·K·(N − K + 1) FLOPs and
(2N + 1)·s bytes.

| Case | dtype | AI (FLOP/B) | FLOPs | Bytes | t_compute @ 60.8 | t_DRAM @ 947 / 802 GB/s | Bound |
|---|---|---:|---:|---:|---:|---:|---|
| N = 2²⁰, K = 31 | fp32 | 7.75 | 65.0 M | 8.39 MB | 1.1 µs | 8.9 / 10.5 µs | **L2-resident:** launch and L2 |
| N = 25M, K = 31 | fp32 | 7.75 | 1.55 G | 200 MB | 25.5 µs | 211 / 249 µs | DRAM, 8× |
| N = 25M, K = 31 | fp16 | 15.5 | 1.55 G | 100 MB | 25.5 µs | 106 / 125 µs | DRAM, 4× |
| N = 25M, K = 255 | fp32 | 63.75 | 12.75 G | 200 MB | 209.7 µs | 211 / 249 µs | **On the ridge** |
| N = 25M, K = 255 | fp16 | 127.5 | 12.75 G | 100 MB | 209.7 µs | 106 / 125 µs | Compute, 2× |

- **FP32 K = 255 sits on the ridge**: 209.7 µs of FFMA against 211 µs of DRAM at peak.
  Against achieved ceilings (218 vs 249 µs) it is mildly memory-bound. A kernel at the
  roofline would show high DRAM and FMA utilization together, which makes this the most
  interesting point in the study.
- **N = 2²⁰ never reaches DRAM between samples.** Its 8 MiB footprint fits in the 64 MiB L2,
  and `lab.bench` reuses inputs. It is a launch- and L2-bound case, not the DRAM-bound
  side of the ridge, so `BENCH_CASES` adds N = 25,000,000 with K = 31 as the DRAM-bound
  counterpart.
- **FP16/BF16 halve the bytes but not the FLOPs** (FP32 accumulation means FFMA plus one
  conversion per loaded element), so K = 255 becomes compute-bound. HFMA2 would double the
  FLOP rate, but it accumulates in reduced precision and the tolerances reject it (§2).
  Tensor cores, via a Toeplitz GEMM, are outside the contract's arithmetic and out of scope.

### The naive kernel: the LSU/L1 pipe, not DRAM (main hypothesis)

A one-thread-per-output kernel issues, per tap, one FFMA and two loads: `x[i + j]`
(coalesced, but misaligned by `j` elements, so a warp's 128 B usually spans two 128-B lines)
and `w[j]` (one address per warp, a broadcast). Neighbouring threads and warps reuse each
`x` element K times, so L1 and L2 absorb almost all of it. **DRAM traffic stays near the
2s-bytes-per-output minimum; what scales with K is load instructions and L1 wavefronts.**

Assumed issue rates per SM per clock (to be confirmed with ncu): 4 warp-wide FFMAs (128
lanes), and about 1 L1/LSU data wavefront. With 2–3 wavefronts per tap (1–2 for `x`, 1 for
`w`), the naive kernel is capped at 1/8–1/12 of FFMA peak, **about 5–7.6 TFLOP/s**:

| Case (fp32) | Roofline bound | Naive LSU bound | Naive prediction |
|---|---:|---:|---|
| N = 25M, K = 31 | 249 µs (DRAM) | 204–304 µs | About equal: already near the LSU limit at 8 FLOP/B |
| N = 25M, K = 255 | 249 µs (DRAM ≈ FMA) | 1.7–2.5 ms | **7–10× off the roofline, LSU-bound** |

Corollaries to test:

1. **The naive kernel moves near-minimum DRAM bytes at every K.**
   `dram__bytes_read.sum` per output ≈ s, and the L1 hit rate is high.
2. **Its limiter is the LSU/L1 pipe.** L1 throughput (`l1tex__throughput`,
   `l1tex__data_pipe_lsu_wavefronts`) near peak, FMA pipe utilization
   (`sm__pipe_fma_cycles_active`) low, and stalls dominated by MIO/LG throttle, not
   long scoreboard (DRAM latency).
3. **A shared-memory tile by itself barely helps.** LDS goes through the same MIO/L1 data
   path, so one LDS per tap replaces one LDG per tap. It removes tag lookups and the
   line-crossing misalignment (2 wavefronts → 1), so expect up to about 1.5–2×, not the 7–10×
   gap. Shared memory's job here is to feed registers with aligned, conflict-free reads.
4. **Register blocking closes the gap.** If each thread computes R consecutive outputs and
   slides its window through registers, one `x` load feeds R FFMAs. With `w` held as a
   warp-uniform operand (uniform registers or `__constant__`, where FFMA reads it straight
   from the constant bank), loads per FFMA drop to 1/R. R ≥ 8 should take the
   FMA:wavefront ratio past 4:1 and make K = 255 FFMA- or DRAM-bound.

Two caveats for the profiling step. Nsight Compute locks clocks to base by default, so read
its utilization percentages, not its durations, against the 2827 MHz ceilings above. Also,
the FFMA peak assumes FFMA is the only instruction: address arithmetic, conversions and
loop overhead must stay small relative to R·K FFMAs per thread.

## 4. Triton: the naive stencil, measured

`triton/implementation.py` is the per-output stencil from §3. Each program owns BLOCK
consecutive outputs in FP32 accumulators. Per tap it loads `w[j]` (one address) and the
window `x[offs + j]`, then does one FMA per output. Triton spreads the BLOCK outputs over
`32 · num_warps` threads, so each thread owns `E = BLOCK / (32 · num_warps)` consecutive
outputs. The window starts at an arbitrary `j`, so the compiler cannot prove alignment and
emits E scalar loads per tap, not one vector load.

### The hypothesis holds: LSU/L1-bound, DRAM idle

Nsight Compute, FP32, N = 25,000,000, with the originally submitted tile (BLOCK = 1024,
4 warps, E = 8):

| Metric | K = 31 | K = 255 |
|---|---:|---:|
| L1 throughput (% of peak) | 98.1 | 99.7 |
| Warp stalls on LG throttle (%) | 67.8 | 69.5 |
| FMA pipe active (%) | 11.6 | 12.3 |
| L1 hit rate (%) | 98.4 | 99.7 |
| L2 ↔ L1 bytes (minimum 200 MB) | 230 MB | 289 MB |
| DRAM throughput (% of peak) | 38.8 | 5.0 |

DRAM traffic stays near its minimum while L1 is saturated and the FMA pipe is mostly idle.
Precision makes no difference: FP16 and BF16 take the same time as FP32 despite moving
half the bytes. This holds even at K = 31, which the roofline calls DRAM-bound: the naive
kernel takes about 450 µs against a 249 µs DRAM floor.

### What actually saturates: requests and wavefronts

A tile sweep (K = 255, FP32) found that one element per thread, the fully coalesced
layout, is *not* the fastest. Seven Nsight Compute profiles fit a two-resource model to
about 1%. At 2.70 GHz, L1 accepts about **0.50 load requests** (`l1tex__lsuin_requests`)
and returns about **1.0 data wavefront** per SM per clock:

| dtype | E | bytes/thread | requests (M) | wavefronts (M) | wavefronts/request | limiter | ncu time |
|---|---:|---:|---:|---:|---:|---|---:|
| fp32 | 1 | 4 | 398 | 435 | 1.09 | requests (99.9%) | 3.54 ms |
| fp32 | 2 | 8 | 299 | 520 | 1.74 | requests | 2.66 ms |
| fp32 | 4 | 16 | 249 | 860 | 3.45 | wavefronts | 3.81 ms |
| fp32 | 8 | 32 | 224 | 828 | 3.69 | wavefronts (99.7%) | 3.68 ms |
| fp16 | 2 | 4 | 299 | 308 | 1.03 | requests | 2.66 ms |
| fp16 | 4 | 8 | 249 | 456 | 1.83 | both (99.7% / 90.9%) | 2.22 ms |
| fp16 | 8 | 16 | 224 | 827 | 3.69 | wavefronts | 3.67 ms |

Per 32·E outputs and per tap, a warp issues E requests for `x` and one for `w`, so more
outputs per thread means fewer requests. But a warp's scalar load spans 32·E·s bytes, so
wavefronts per request grow with bytes per thread. **8 bytes per thread balances the two**,
so the default tile is `32 · num_warps · 8/s`: 256 for FP32 and 512 for FP16/BF16 at
4 warps. That is 1.4× (FP32) and 1.65× (FP16/BF16) faster than the original 1024 × 4 tile.

The balanced FP16 tile runs at its request limit: 249M requests / (84 SMs · 0.5/clk ·
2.827 GHz) = 2.10 ms, against 2.12 ms measured. **No tile choice can take this design
past about 6 TFLOP/s.** That is the 5–7.6 TFLOP/s band §3 predicted, 8–10× below the
roofline at K = 255. Closing the gap needs fewer L1 requests per FMA, which means register
reuse of each loaded `x` across several outputs (§3, corollary 4). Triton exposes no
register shifts for this pattern, so the plan is to test that in the CUDA variants.

### Benchmark (default tile)

`lab.bench`, 2026-11-03, RTX 5080 (cuda.0; clocks unlocked, no other compute processes),
Triton 3.8.0, PyTorch 2.14.1, cuDNN 9.24. Each value is the median of three run medians,
each taken from CUDA-event samples with inputs reused and output allocation included.
The run-median spread is 0.0–3.8% for Triton. `copy` is a device copy of `x`, the DRAM
ceiling. Raw JSON and sanitizer logs are in `artifacts/convolution_1d/`.

| Case | dtype | triton µs | TFLOP/s | pytorch µs | copy µs | Roofline floor µs |
|---|---|---:|---:|---:|---:|---:|
| N = 2²⁰, K = 31 | fp32 | 25.5 | 2.55 | 137.3 | 9.2 | (L2-resident) |
| | fp16 | 25.1 | 2.59 | 40.9 | 8.9 | |
| | bf16 | 23.5 | 2.76 | 39.6 | 9.0 | |
| N = 25M, K = 31 | fp32 | 334.2 | 4.64 | 3,034 | 245.3 | 249 (DRAM) |
| | fp16 | 286.1 | 5.42 | 728 | 123.5 | 125 (DRAM) |
| | bf16 | 281.2 | 5.51 | 725 | 123.7 | 125 (DRAM) |
| N = 25M, K = 255 | fp32 | 2,533.8 | 5.03 | 19,673 | 246.0 | 249 (DRAM ≈ FMA) |
| | fp16 | 2,128.4 | 5.99 | 860,601 | 123.5 | 218 (FMA) |
| | bf16 | 2,119.7 | 6.01 | 3,683 | 123.5 | 218 (FMA) |

Floors use the achieved ceilings (802 GB/s, 58.6 TFLOP/s). At K = 255, Triton is 10.2×
(FP32) and 9.7× (FP16/BF16) above the floor. At K = 31 from DRAM it is 1.3× (FP32) and
2.3× (FP16/BF16) above. Against cuDNN it is 1.6–9.1× faster, and 404× faster for the FP16
K = 255 fallback.

### The PyTorch baseline is not a strong competitor

cuDNN picks a different algorithm for each dtype on this one-channel shape, with or
without `cudnn.benchmark`:

| dtype | K = 31 | K = 255 | kernel |
|---|---:|---:|---|
| fp32 | 2.99 ms | 19.4 ms | `implicit_convolve_sgemm` |
| fp16 | 0.72 ms | **858 ms** | direct kernel at K = 31; `conv1D_NCHW_general` fallback at K = 255 |
| bf16 | 0.72 ms | 3.63 ms | `conv2d_grouped_direct_kernel` |

The FP16, K = 255 reference therefore takes almost a second per call. That is acceptable
for a correctness reference but makes it a poor performance baseline: compare kernels
with the roofline and with each other.

## 5. CUDA variants

Both kernels use one thread per output, 256 threads per block, FP32 accumulation with
`fmaf`, and a single rounding to the input dtype. `cuda` is an alias for `cuda_naive`,
the CUDA kernel that accepts every K.

| Backend | Kernel | Hypothesis |
|---|---|---|
| `cuda_naive` (`cuda`) | `x` and `w` read from global memory | L1 hit rate is high, but the kernel is limited by LSU/L1. Per tap a warp issues 2 global loads: a coalesced, `j`-misaligned `x` load and a broadcast `w` load. That is 2K load instructions per 32 outputs, against the 2s-byte DRAM minimum per output |
| `cuda_const` | `w` converted to FP32 and staged in `__constant__` memory | Every lane reads the same tap, so the constant cache broadcasts it and the `w` loads leave the LSU/L1 pipe. Requests per tap fall from 2 to 1, so up to 2× faster if requests are the limit (§4) |

**Constant-memory details.** The filter is stored as FP32, so the capacity is the whole
64 KiB user constant bank, 16384 taps, for every dtype. FP16/BF16 filters stored natively
would fit 32768 taps but would need a conversion per tap. Each call converts `w` to FP32
(no copy for FP32) and uploads it with `cudaMemcpyToSymbolAsync` on the current stream,
before the kernel. One buffer per device serves every call, so calls on one stream are
safe, but concurrent `cuda_const` calls on different streams of one device would race.
K > 16384 is rejected with `ValueError` rather than silently falling back.
`test_cuda_const_reloads_filter` covers back-to-back calls with long and short filters
and no host synchronization in between.
