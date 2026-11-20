# Design

[← Gaussian Blur](../README.md) · **Design**

## 0. Source verification

**Verified** against the LeetGPU statement and checker in the public challenge repository,
[`AlphaGPU/leetgpu-challenges/challenges/medium/28_gaussian_blur`](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/28_gaussian_blur)
(`challenge.html`, `challenge.py`, `starter/starter.cu`; last commit 7691ca5, 2026-10-11),
read on 2026-11-17. The [challenge page](https://leetgpu.com/challenges/gaussian-blur) itself
renders client-side. No submission was made.

| Question | Source answer |
|---|---|
| Kernel argument | **A 2-D array** `kernel` (row-major FP32) with `kernel_rows × kernel_cols`, not σ and not a 1-D kernel |
| Square or rectangular | **Rectangular**: `kernel_rows` and `kernel_cols` are separate, both odd, 3 to 21 |
| Boundary | **Zero padding** ("treat values outside the image boundary as zeros"), not clamp |
| Output | **Same size** as the input, written to a caller-supplied `output` |
| Orientation | `input[i+m, j+n] × kernel[m+kh/2, n+kw/2]`: cross-correlation, no flip |
| Image size | 1 ≤ rows, cols ≤ 4096; FP32 only |
| Performance case | **512 × 512 image, 7 × 7 kernel**; image uniform in [0, 255), kernel **uniform in [0.0001, 0.02)** |
| Checker | `F.conv2d(input, kernel, padding=(kh//2, kw//2))`, `atol = rtol = 1e-5` |

Signature: `solve(const float* input, const float* kernel, float* output, int input_rows,
int input_cols, int kernel_rows, int kernel_cols)`. The lab uses an allocating tensor API
instead.

**The source's own data contradicts its constraints.** It says all kernel values are
non-negative and sum to 1, but:

- The performance kernel is 49 independent uniform values summing to about 0.5, and its
  rank-1 residual is 39% of its norm ([§2](#separability)). **The timed case is neither
  normalized nor separable.**
- The `large_random` functional test (32 × 32, 5 × 5, uniform [0, 1)) is unnormalized and
  non-separable too. So is example 2 (`0.1` with `0.2` at the centre, σ₂/σ₁ = 0.17).
- Only example 1 (the binomial 1-2-1 outer product), the identity, and the 0.111111 box are
  separable.
- The `single_pixel` test uses a 1 × 1 kernel, below the stated minimum of 3.

The answer to step 0 is therefore the first branch: the kernel arrives as a 2-D array, and
whether it is separable has to be **detected**. A backend that assumes separability fails
the timed case. It is also why the checker runs `F.conv2d` with PyTorch's default
`cudnn.allow_tf32 = True`; [§2](#2-correctness-reference) explains why the lab reference
does not.

## 1. Algorithm and contract

```text
out[i, j] = Σ_{m<KH} Σ_{n<KW} image[i + m − KH/2, j + n − KW/2] · kernel[m, n]
            with image[r, c] = 0 outside 0 ≤ r < H, 0 ≤ c < W
```

- **Inputs:** contiguous strided 2-D `image` (H, W) and `kernel` (KH, KW), H, W ≥ 1, KH,
  KW ≥ 1 and odd, both fp32, fp16 or bf16, same device. The kernel may be larger than the
  image; then most of every window is padding. Storage offsets are allowed, so inputs need
  not be 16-byte aligned. With odd W, every row after the first is misaligned anyway.
- **Output:** a fresh contiguous (H, W) tensor, same dtype and device. Inputs are not mutated.
- **Precision:** products are accumulated in FP32 and rounded once to the input dtype.
- **Separable path:** a backend may run a KW-tap row pass and a KH-tap column pass only
  when `api.separable_factors` accepts the kernel. It must use those FP32 factors and keep
  the intermediate in FP32 (in memory or on chip), so the output is still rounded once.
- **Lab widenings beyond the source:** FP16/BF16; KH, KW = 1; kernels past 21 (tests go to
  31 × 31); images past 4096 are allowed but not tested.
- **Rejected by `api.validate`:** other ranks, non-contiguous inputs, mixed dtype or
  device, empty tensors, even kernel sizes, fp64/integer/sparse inputs, `requires_grad`,
  and unresolved negative/conjugate views.

## 2. Correctness reference

`F.conv2d(image[None, None], kernel[None, None], padding=(KH//2, KW//2))`, inside
`torch.backends.cudnn.flags(..., allow_tf32=False)` with the other cuDNN flags unchanged.
The reference always runs the direct 2-D filter, so it is exact to the supplied kernel and
never depends on the separability decision.

**The TF32 trap.** PyTorch defaults to `cudnn.allow_tf32 = True`, and so does the source
checker. A TF32 product has a 10-bit mantissa; emulated, it fails the 1e-5 tolerance by
9–26× ([table below](#tolerances)). If cuDNN ever picked a TF32 algorithm for this shape, the
checker would reject correct FP32 kernels. Whether cuDNN does so for a one-channel conv2d on
the RTX 5080 is still to be measured (it did not for conv1d); the reference turns TF32 off
either way, and `test_reference_matches_oracle` runs with it switched on globally.

### FP64 oracle

`oracle` gathers each zero-padded window with `F.unfold` in FP64 (a pure gather, so exact)
and takes `math.fsum` of the FP64 products. Every input converts to FP64 exactly and an
FP32 × FP32 product is exact in FP64, so each value is the exact sum, correctly rounded. A
direct result must satisfy the componentwise bound for any FP32-accumulated sum of
n = KH·KW products rounded once to dtype `d` (Higham, Thm 3.1; any order, with or without FMA):

```text
|ŷ − y| ≤ u_d·|y| + (1 + u_d)·γ_n·Σ|x·k| + η_d,     γ_n = n·u₃₂ / (1 − n·u₃₂),  n = KH·KW
```

Edge cases in `TEST_CASES`:

- 1 × 1 images with 1 × 1, 3 × 3 and 21 × 21 kernels (the kernel larger than the image).
- A 2 × 37 image under a 21 × 21 kernel, where every window is mostly padding.
- Borders on every case, where up to KH//2 rows and KW//2 columns of a window are padding.
- Odd W (13, 17, 31, 37, 41, 65, 67), so tiles have tails and rows are misaligned.
- 1 × 7 and 7 × 1 kernels, and rectangular 3 × 5 and 5 × 3 kernels.
- 31 × 31, past the source limit.

Maximum error/bound for the CPU reference, seed 1 (cuDNN figures to be added from cuda.0):

| Case (H×W, KH×KW, kernel) | fp32 | fp16 | bf16 |
|---|---:|---:|---:|
| 1×1, 3×3 Gaussian | 0.049 | 0.448 | 0.086 |
| 5×5, 3×3 Gaussian | 0.197 | 0.778 | 0.812 |
| 9×13, 3×5 Gaussian | 0.094 | 0.969 | 0.964 |
| 64×67, 7×7 Gaussian | 0.085 | 0.969 | 0.986 |
| 64×67, 7×7 random | 0.056 | 0.954 | 0.981 |
| 33×65, 21×21 Gaussian | 0.010 | 0.772 | 0.944 |
| 37×41, 31×31 random | 0.002 | 0.486 | 0.809 |

FP16 and BF16 approach 1 because a single output rounding can come close to half an ulp.
The bound is rigorous, so this is not flakiness.

### Separability

`api.separable_factors(kernel)` takes the rank-1 term of the FP64 SVD, σ₁·u₁·v₁ᵀ. It splits
it evenly as `col = √σ₁·u₁`, `row = √σ₁·v₁` (sign fixed so the factors are non-negative for
a blur) and rounds both to FP32. It then computes the residual `R = kernel − col·rowᵀ`
exactly in FP64 against those rounded factors, and accepts when

```text
‖R‖_F ≤ ε_d·‖kernel‖_F,    ε_d = 4·u_d   (fp32 2.4e-7, fp16 2.0e-3, bf16 1.6e-2)
```

This is the σ₂/σ₁ < ε rule, made rigorous. ‖R‖_F also contains σ₃, σ₄, … and the error
of rounding the factors to FP32, which is everything a two-pass backend actually discards.
ε_d is relative to the **input** dtype because the kernel itself arrives rounded to it: a
Gaussian in FP16 is rank-1 only to about u₁₆, so an FP32 threshold would send every FP16
Gaussian down the direct path. Accepting k ≈ col·rowᵀ then perturbs the result by a few
units of the dtype's own roundoff, the size of the output rounding the contract already
allows.

Measured gap (CPU, FP64 SVD), relative residual in units of u_d:

| Kernel | fp32 | fp16 | bf16 | σ₂/σ₁ |
|---|---:|---:|---:|---:|
| Gaussian 3×3 … 31×31, outer product, rounded to d | 0.31–0.74 | 0.00–0.41 | 0.00–0.39 | fp32 ≤ 2.0e-8, bf16 ≤ 1.5e-3 |
| Gaussian sampled from σ in 2-D | accepted (test) | accepted | accepted | |
| Source random, 3×3 … 31×31 | rejected | rejected | rejected | 0.19–0.30 |
| Source example 2 | rejected | rejected | rejected | 0.17 |

Honest separable kernels sit below 1 u_d and the source's kernels sit at 0.2 relative,
10⁵–10⁶ u₃₂ away, so the threshold is not delicate. A Gaussian perturbed by 8 u_d in a
corner tap is rejected (test). A bump on the centre tap mostly lines up with the
Gaussian's own rank-1 direction, so the SVD absorbs it.

**Cost.** The SVD runs on the host: a CUDA kernel is copied back, which synchronizes the
stream. That cost belongs to whoever calls `separable_factors`, once per kernel; it is
measured with the backends.

**Two-pass bound.** A two-pass result sends each product through a KW-term and then a
KH-term FP32 sum, so its summation term counts KH + KW instead of KH·KW. The discarded
residual adds its own term:

```text
|ŷ − y| ≤ u_d·|y| + (1 + u_d)·(γ_{KH+KW}·Σ|x|·|col|·|row|ᵀ + Σ|x|·|R|) + η_d
```

`test_two_pass_matches_oracle` checks it for an emulated two-pass (two cuDNN/CPU 1-D
convolutions, FP32 intermediate) on every Gaussian case and dtype: worst 0.88 (fp16),
which is the final rounding again. `test_two_pass_within_tolerance` checks that factoring
keeps every Gaussian case, and the 4096² × 21² bench case on the GPU, within the fixed
tolerances against the direct reference. Worst ratio 0.052 (bf16, 7 × 7).

### Tolerances

Every backend is compared with the reference under fixed `(rtol, atol)`:

- **fp32 (1e-5, 1e-5).** This matches the source checker, 10× tighter than conv1d.
- **fp16 (1e-2, 1e-2)** and **bf16 (5e-2, 5e-2).**

Two correct implementations differ in three ways:

1. **Final rounding.** At most 1 ulp of the output, absorbed by `rtol` and independent of K.
2. **FP32 summation order.** Noise grows with the term count: KH·KW for direct, KH + KW
   for two passes.
3. **Factorization residual,** separable path only. It is bounded by ε_d·‖x_window‖·‖k‖.

Measured as max |a − b| / (atol + rtol·|b|), where above 1 fails (CPU reference, 96 × 131
image):

| Check | fp32 | fp16 | bf16 |
|---|---:|---:|---:|
| Running FP32 sum over all taps, row-major and reverse, 21×21 / 31×31 Gaussian | 0.030 / 0.027 | 0.011 / 0.011 | 0.000 / 0.009 |
| Same, 21×21 / 31×31 random | 0.059 / 0.101 | 0.010 / 0.017 | 0.002 / 0.009 |
| Two passes (factored) vs direct, every Gaussian case | ≤ 0.018 | ≤ 0.031 | ≤ 0.052 |
| **Negative:** TF32 products, 7×7 / 21×21 | 21.1–26.1 / 8.7–9.9 | | |
| **Negative:** FP16/BF16 accumulation, 7×7, unit-normal image | | 0.12 | 0.16 |
| **Negative:** same, 7×7, image in [0, 255) | | 0.42 | 0.62 |
| **Negative:** same, 21×21, image in [0, 255) | | 1.56 | 2.07 |

**Verdict.** The fixed tolerances hold with at least 10× headroom up to 31 × 31 (961 terms,
fewer than conv1d's 2047), so they are not scaled with K.

**But a fixed tolerance cannot reject FP16/BF16 accumulation for a blur.** A normalized
kernel keeps partial sums small, so accumulating in FP16 costs about as much as the final
rounding. At the source's 7 × 7 kernel it passes the tolerance (0.12–0.62); it only fails
at 21 × 21 on [0, 255) data. So the kernel-scaled check is the one that does the work:
`test_matches_oracle` holds **every backend** to `contract_bound`. That is the FP64 bound
above with n = KH·KW, widened to the two-pass bound when the kernel is separable. It
tightens for small kernels and grows for large ones.

`test_bound_rejects_reduced_precision`, 7 × 7, unit-normal image (contract-bound ratios,
above 1 fails):

| Path | fp32 | fp16 | bf16 |
|---|---:|---:|---:|
| TF32 products / FP16 / BF16 accumulation, Gaussian | 86 | 10.4 | 3.3 |
| Same, random kernel | 97 | 84 | 548 |
| Reference, Gaussian / random | 0.06 / 0.05 | 0.88 / 0.96 | 0.57 / 0.99 |
| Two passes, Gaussian | 0.04 | 0.88 | 0.77 |

Exact checks need no tolerance: `test_reference_contract` uses small-integer data, so FP32
partial sums are exact and the result must equal the oracle bit for bit, and
`test_source_examples` reproduces example 1 exactly (dyadic kernel).
