# 09 · 1D Convolution

Valid 1-D cross-correlation, `out[i] = Σ_j x[i + j]·w[j]`, accumulated in FP32. Each output
reads K neighbours, so neighbouring outputs share almost all their inputs, and the study's
primary lesson is **halo reuse in shared memory**: when staging a tile and its halo pays, and
what it actually relieves. Arithmetic intensity grows with the filter (about K/e FLOP/B), so
one kernel sweeps from DRAM-bound to compute-bound across the RTX 5080's ridge at K ≈ 294
(FP32).

**Status:** PyTorch, CUDA (6 variants) and Triton implemented, validated and measured on RTX
5080. 622 GPU correctness tests passed (60 expected skips); Compute Sanitizer memcheck,
racecheck, synccheck and initcheck: 0 errors, 0 hazards. Source: [LeetGPU 1D
Convolution](https://leetgpu.com/challenges/1d-convolution) (statement not verified: the page
renders client-side; see [design §0](docs/design.md#0-source-verification)).

![Top: achieved TFLOP/s against K under the measured roofline; naive and cuda_smem flatten at 3.8 TFLOP/s, Triton at 5.0, cuda_const at 7.4, while the register-window kernels climb to 23-28 TFLOP/s past the ridge at K ≈ 294. Bottom: effective GB/s against the 802 GB/s copy ceiling.](figures/roofline_vs_k.svg)

## Key results

- **The bottleneck is L1 requests, not DRAM.** For every kernel and every K, DRAM reads stay
  at one element per output; L1/L2 absorb the K-fold reuse and every halo. One-output-per-
  thread kernels leave the roofline at K = 15–31 onto a K-independent plateau set by load
  requests per FMA: 3.8 TFLOP/s for `cuda_naive` (2 requests per warp-FMA), 7.4 for
  `cuda_const` (1 request: its filter moves to uniform registers, exactly 2× naive).
- **Shared memory alone never pays.** `cuda_smem` is slower than naive at small K (323 vs
  290 µs, FP32 K = 31) and equal at large K: one shared load replaces one global load, the
  requests per FMA do not change, and the barriers cost extra. L1 already caught the reuse
  (95–99% hit rate).
- **Shared memory plus registers does: 6–7×.** `cuda_smem_reg` (8 outputs per thread from a
  sliding register window, conflict-free FP32 tile) reaches 23.0 TFLOP/s at K = 2047;
  `cuda_large_k` (filter also in shared memory) 27.7; R = 16 29.8, about half the 58.9 TFLOP/s
  FFMA peak. The new limit is still LSU request intake (0.26 requests per warp-FMA, 87% busy)
  and issue slots.
- **Halo overhead follows (K − 1)/span exactly** (to 0.3%, spans 512–4096, K 31–2047), and
  none of it reaches DRAM.
- **Short filters:** at K ≤ 15 `cuda_naive`, `cuda_const` and Triton run at the 802 GB/s copy
  ceiling; the register-window kernels are 25–30% slower there. `cuda_unrolled` (K = 31) is
  the fastest K = 31 kernel at 93% of the ceiling.
- **Almost no L2 cliff at K = 31:** the copy ceiling drops from 2280 to 802 GB/s when the
  data leaves L2, but only the two kernels nearest the ceiling lose anything (`cuda_const`
  and `cuda_unrolled`, 10–13%); the rest are not bandwidth-bound even in L2.
- **cuDNN is a weak baseline:** 12× slower than the best CUDA kernel at FP32 K = 31, and its
  FP16 path falls back to a generic kernel from K = 255 (861 ms at N = 25 M).

Recommended: `cuda_large_k` for K ≥ 127, `cuda_smem_reg` (the `cuda` alias) as the general
choice from K ≈ 63, `cuda_const` or `cuda_unrolled` below. A K-based dispatch would beat the
alias everywhere; see [results](docs/results.md#other-findings).

## Documentation

| Page | Contents |
|---|---|
| [Design](docs/design.md) | Contract, reference and tolerances, performance model, Triton analysis, CUDA variants and hypotheses |
| [Results](docs/results.md) | K, N and R sweeps, Nsight Compute profiles, halo model, answers and limitations |
| [Testing](docs/testing.md) | FP64 oracle, tolerance controls, boundary suites, sanitizers |
| [Benchmarking](docs/benchmarking.md) | Reproduce the sweeps and what one sample measures |
| [Profiling](docs/profiling.md) | Nsight Compute reports, metrics and the question each answers |

## Quick start

```bash
python -m pytest -q problems/convolution_1d
PYTHON=python bash problems/convolution_1d/benchmarks/sweep.sh
```

```python
import torch
from problems.convolution_1d.api import conv1d

x, w = torch.randn(1 << 24, device="cuda"), torch.randn(255, device="cuda")
out = conv1d(x, w, backend="cuda_large_k")  # out[i] = sum_j x[i + j] * w[j], N - K + 1 outputs
```

Backends: `pytorch` (`F.conv1d`, TF32 off), `cuda_naive`, `cuda_const` (K ≤ 16384),
`cuda_smem`, `cuda_smem_reg` (the `cuda` alias), `cuda_unrolled` (K ∈ {3, 7, 31}),
`cuda_large_k` and `triton`. `cuda_smem_reg` and `cuda_large_k` take `reg_outputs` ∈ {2, 4, 8,
16}; Triton takes `block_size` and `num_warps`. FP32, FP16 and BF16 inputs, always
accumulated in FP32. The JAX baseline is not implemented.

```text
convolution_1d/
    api.py         validated public contract
    cases.py       test and benchmark inputs, bytes and FLOPs
    pytorch/       F.conv1d reference (TF32 off)
    cuda/          kernels.cu, bindings.cpp, implementation.py (six variants)
    triton/        direct-stencil kernel, 8 bytes of outputs per thread
    jax/           JAX baseline (not implemented)
    tests/         test_convolution_1d.py
    benchmarks/    sweep, sweep driver, FFMA peak, capture, summary and plot
    docs/          design, results, testing, benchmarking, profiling
```

Raw benchmark JSON, 78 Nsight Compute reports with CSV exports, sanitizer logs and
disassembly are in `artifacts/convolution_1d/` in this workspace. Large
generated files are ignored by git; [compact evidence](benchmarks/measured_summary.json), the
scripts and the figures are tracked.

Prerequisites: [08 · Matrix Transpose](../matrix_transpose/README.md). Follow the
[optimization methodology](../../docs/methodology.md).
