# 05 · RGB to Grayscale

`gray = 0.299·R + 0.587·G + 0.114·B` over an image of interleaved RGB pixels: three reads
per write, 12 bytes per pixel in FP32. A memory-bound kernel for studying whether a
three-channel stride costs anything, and what wider accesses, shared-memory staging or a
planar layout buy.

**Status:** 🟨 CUDA and PyTorch complete, Triton not implemented yet · validated and measured
on NVIDIA GeForce RTX 5080 · primary lesson: **interleaved channel access** · prerequisite:
[04 · Interleave Arrays](../interleave_arrays/README.md).

![Effective bandwidth of every backend at 2048 × 2048 and 4096 × 4096 on RTX 5080. At 4096 × 4096 the CUDA kernels and torch.compile reach 809–866 GB/s, below the 960 GB/s DRAM peak; at 2048 × 2048 FP16 cuda_vec reaches 2060 GB/s because the image stays in L2.](figures/rgb_bandwidth.svg)

## Key results

RTX 5080. 4096 × 4096 (a lab extension) is DRAM-bound; the source's 2048 × 2048 FP16 image
stays in L2. Details are in [results](docs/results.md).

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
#           cuda_pixel, cuda_vec, cuda_smem, cuda_planar
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
├── triton/          stub: not implemented yet
├── tests/           source examples, channel order, misalignment, every backend vs the reference
├── benchmarks/      sweep.sh (benchmarks + Nsight reports), plot.py
├── figures/         generated chart (tracked)
└── docs/            design, results, testing, benchmarking, profiling
```
