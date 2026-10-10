# 03 · Reverse Array

`x[i] ↔ x[N−1−i]`, in place: a pure data-movement kernel for studying race-free ownership,
coalescing in both directions, and how alignment decides whether wide memory accesses
are possible.

**Status:** ✅ Complete · validated and measured on NVIDIA GeForce RTX 5080 ·
primary lesson: **race-free in-place indexing** · prerequisite:
[01 · Vector Addition](../vector_add/README.md).

![Reverse Array speed relative to a device copy at N = 100M on RTX 5080, for aligned and misaligned N. FP32: every custom kernel reaches 97–99.6% of copy speed and PyTorch about 51%. FP16: cuda_tile, cuda_tile_async, triton_flip and triton_pair reach 98–99.6% for both N; cuda_vec drops from 99% to 84% when N is misaligned; cuda_pair reaches 73–91%.](figures/reverse_vs_copy.svg)

## Key results

RTX 5080, N = 100M (DRAM-bound). A device copy moves the same bytes, so it is the
practical speed limit. Details are in [results](docs/results.md).

- **Reverse order is free; alignment is not.** A warp reading the mirrored half in
  descending order touches the ideal 4 sectors (FP32); misaligned N adds exactly one.
- **`cuda_tile` is the best kernel: 98–99.6% of copy speed for every dtype and N.** It
  stages each block's front and back ranges through shared memory, so 16-byte accesses
  work even when the mirrored half is misaligned.
- **Wide accesses matter for 2-byte types.** One-element-per-thread `cuda_pair` reaches
  only 73–91% in FP16; 16-byte variants reach 98–99.6%.
- **PyTorch takes twice as long as a copy,** because `torch.flip` writes a reversed copy
  that is then copied back.
- **Triton matches CUDA `tile` at DRAM-bound sizes,** although for misaligned N it
  falls back to scalar accesses that touch 4–8× the needed sectors per instruction.

## Documentation

| Page | Contents |
|---|---|
| [Design](docs/design.md) | Contract, ownership rule, alignment math, performance model, the variants |
| [Results](docs/results.md) | Latency per backend, profiling (why the variants differ), Triton, limitations |
| [Testing](docs/testing.md) | Test cases, running the suite, Compute Sanitizer |
| [Benchmarking](docs/benchmarking.md) | Commands, copy reference, benchmark sizes, reproduction |
| [Profiling](docs/profiling.md) | Capturing one launch, report sections, metrics and how to read them |

## Quick start

From the repository root, after [setup](../vector_add/docs/testing.md#setup):

```bash
python -m pytest -q problems/reverse_array
python -m lab.bench reverse_array --copy-reference --peak-gb-s 960
```

```python
import torch

from problems.reverse_array.api import reverse_

x = torch.arange(10, dtype=torch.float32, device="cuda")
# Backends: pytorch, cuda_pair, cuda_vec, cuda_tile, cuda_tile_async,
# triton_pair, triton_flip
reverse_(x, backend="cuda_tile")  # x is now [9, 8, ..., 0]
```

## Layout

```text
reverse_array/
├── README.md        overview (this page)
├── api.py           public entry point, contract, backends and tolerances
├── cases.py         sizes (odd/even, boundaries, N mod 8) and value ranges
├── pytorch/         reference: torch.flip + copy_
├── cuda/            pair, vec, tile and tile_async kernels, bindings, loader
├── triton/          pair and flip kernels
├── tests/           source examples, in-place semantics, every backend vs the reference
├── benchmarks/      sweep.sh (benchmarks + Nsight reports), plot.py
├── figures/         generated chart (tracked)
└── docs/            design, results, testing, benchmarking, profiling
```
