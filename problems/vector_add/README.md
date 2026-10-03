# 01 · Vector Addition

`C[i] = A[i] + B[i]`: a first streaming-kernel study. It separates launch cost from
memory throughput, then tests what vectorized access and grid-stride loops each change.

**Status:** ✅ Complete · validated and measured on NVIDIA GeForce RTX 5080 ·
primary lesson: **launch overhead vs bandwidth** · no prerequisites.

![Effective bandwidth vs working-set size on RTX 5080 for PyTorch, CUDA scalar, CUDA vector and Triton, FP32 and FP16. Bandwidth rises linearly with size while launch-bound, peaks near 3 TB/s while the working set fits in the 64 MiB L2, and settles near 840 GB/s once it streams from DRAM.](figures/bandwidth_vs_size.svg)

## Key results

RTX 5080, spec peak 960 GB/s. Full tables, profiler evidence and caveats are in
[results](docs/results.md).

- **Three regimes.** Below ~1 MB, each call costs ~8.3 µs regardless of size
  (launch-bound; Triton ~13.7 µs). While A + B + C fits in the 64 MiB L2, effective
  bandwidth reaches ~3 TB/s. Beyond it, every backend streams from DRAM at
  816–863 GB/s at N = 25M, which is 85–90% of the spec peak.
- **16-byte vector access matters for 2-byte types.** At N = 25M FP16, `cuda_vec`
  takes 173.7 µs vs 183.8 µs for `cuda_scalar` (−5.5%) and 177.6 µs for PyTorch.
  Instructions fall 72% and DRAM throughput rises from 88.5% to 92.6%.
- **For FP32, the grid-stride loop matters more than vectorization.** Vectorization
  changes latency by about ±1%, while adding the capped grid-stride loop costs up to
  2.6%. Plain `cuda_scalar` is the fastest FP32 variant (354.8 µs, 88% of peak).

## Documentation

| Page | Contents |
|---|---|
| [Design](docs/design.md) | Source problem, API contract, the four CUDA kernel variants, performance model |
| [Results](docs/results.md) | Latency per backend, why the kernels differ, small-vector cost, validation, limitations |
| [Testing](docs/testing.md) | Toolchain setup, correctness suite, Compute Sanitizer |
| [Benchmarking](docs/benchmarking.md) | Timing scopes, commands, full reproduction sweep, output format |
| [Profiling](docs/profiling.md) | Nsight Compute, Nsight Systems, register and SASS inspection |

## Quick start

From the repository root, with Python 3.11+ and (for GPU backends) a CUDA-enabled
PyTorch on an NVIDIA host. See [testing](docs/testing.md#setup) for toolchain details.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q problems/vector_add
```

```python
import torch

from problems.vector_add.api import add

a = torch.randn(1025, device="cuda")
b = torch.randn_like(a)
# Backends: pytorch, cuda_scalar, cuda_grid_stride, cuda_vec, cuda_vec_grid_stride, triton
c = add(a, b, backend="cuda_vec")
torch.cuda.synchronize()
```

## Layout

```text
vector_add/
├── README.md            overview (this page)
├── api.py               public, validated entry point: add(a, b, backend=..., out=None)
├── cuda/                kernels.cu, bindings.cpp, lazy extension loader
├── triton/              masked-tile Triton kernel
├── pytorch/             eager baseline
├── jax/                 JAX baseline (native arrays)
├── tests/               correctness and benchmark-statistics tests
├── benchmarks/          custom runner (run.py), plot.py, sweep.sh
├── figures/             generated charts (tracked)
└── docs/                design, results, testing, benchmarking, profiling
```
