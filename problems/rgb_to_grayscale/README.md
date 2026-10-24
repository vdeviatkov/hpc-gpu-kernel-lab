# 05 · RGB to Grayscale

`gray = 0.299·R + 0.587·G + 0.114·B` over an image of interleaved RGB pixels: three reads
per write, 12 bytes per pixel in FP32. A memory-bound kernel for studying whether a
three-channel stride costs anything, and what wider accesses, shared-memory staging or a
planar layout buy.

**Status:** ✅ CUDA, PyTorch and Triton implemented · validated and measured
on NVIDIA GeForce RTX 5080 · primary lesson: **interleaved channel access** · prerequisite:
[04 · Interleave Arrays](../interleave_arrays/README.md).

![Updated three-run RGB comparison including Triton at 2048² and 4096²: Triton reaches 836 GB/s FP32 and 817 GB/s FP16 at 4096².](figures/rgb_bandwidth.svg)

The figure and benchmark tables include three matched runs of every backend,
including Triton (2026-10-23 PDT / 2026-10-24 UTC). Earlier Nsight findings are
labeled separately in [results](docs/results.md).

## Key results

RTX 5080. 4096 × 4096 (a lab extension) is DRAM-bound; the source's 2048 × 2048 FP16 image
stays in L2. Details are in [results](docs/results.md).

- **Triton is competitive for large images:** 320.9 µs FP32 and 164.3 µs FP16
  at 4096². At cache-resident 2048² FP16 it takes 21.4 µs versus 16.3 µs for
  `cuda_vec`. All are complete-call medians from the same session.
- **The stride costs nothing at DRAM-bound sizes.** One pixel per thread (`cuda_pixel`),
  16-byte packs (`cuda_vec`) and shared-memory staging (`cuda_smem`) all read exactly the
  image once at 94% DRAM throughput, 262–265 µs per FP32 kernel in the profiler. L1
  absorbs the stride-3 loads: the second and third channel loads hit the sectors the first
  one fetched.
- **From L2, wide accesses win:** at 2048² FP16 `cuda_vec` takes 7.7 µs (91% of L2
  bandwidth), `cuda_pixel` 14.0 µs, and `cuda_smem` 20.2 µs, slowed by 6.5× the
  instructions, a barrier and 2-way bank conflicts.
- **A planar layout does not help:** the planar kernel alone is as fast as the interleaved
  ones, and converting makes `cuda_planar` 2.6× slower.
- **`torch.compile` matches the CUDA kernels' GPU time** (one fused Triton kernel, 2 pixels
  per thread); it is slower per call (19.5 µs fixed cost against 11 µs). The eager
  reference reads the image three times (4.5–12× slower); `image @ w` runs an
  instruction-bound cuBLAS GEMV (1.8–2.9× slower than the CUDA kernels).

## Documentation

| Page | Contents |
|---|---|
| [Design](docs/design.md) | Contract, the layout problem, performance model, the variants |
| [Results](docs/results.md) | Latency per backend, profiling (why the variants differ), decisions, limitations |
| [Testing](docs/testing.md) | Test cases, running the suite, Compute Sanitizer |
| [Benchmarking](docs/benchmarking.md) | Commands, call cost vs kernel time, sizes, reproduction |
| [Profiling](docs/profiling.md) | Capturing one launch, metrics with definitions and decisions |

## Quick start

From the repository root, after [setup](../vector_add/docs/testing.md#setup):

```bash
python -m pytest -q problems/rgb_to_grayscale
python -m lab.bench rgb_to_grayscale --copy-reference --peak-gb-s 960
```

```python
import torch

from problems.rgb_to_grayscale.api import rgb_to_grayscale

image = torch.tensor([[[255.0, 0.0, 0.0], [100.0, 150.0, 200.0]]], device="cuda")
# Backends: pytorch, pytorch_matmul, pytorch_compile,
#           cuda_pixel, cuda_vec, cuda_smem, cuda_planar, triton
rgb_to_grayscale(image, backend="cuda_vec")  # [[76.245, 140.75]]
```

## Layout

```text
rgb_to_grayscale/
├── README.md        overview (this page)
├── api.py           public entry point, contract, backends and tolerances
├── cases.py         image sizes (every pixel count mod 8) and value distributions
├── pytorch/         reference (the formula, eager), matmul and torch.compile baselines
├── cuda/            pixel, vec, smem and planar kernels, bindings, loader
├── triton/          fused pixel kernel with FP32 arithmetic
├── tests/           source examples, channel order, misalignment, every backend vs the reference
├── benchmarks/      sweep.sh (benchmarks + Nsight reports), plot.py
├── figures/         generated chart (tracked)
└── docs/            design, results, testing, benchmarking, profiling
```
