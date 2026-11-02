# 08 · Matrix Transpose

Transpose an M × N matrix, bit for bit. A transpose moves exactly the bytes of a copy, but
one side of every naive access pattern is strided, so the fix (a shared-memory tile) brings
in the primary lesson: **shared-memory bank conflicts**, and padding or swizzling to remove
them. [07 · Matrix Copy](../matrix_copy/README.md) is the control experiment.

**Status:** PyTorch, CUDA (6 variants) and Triton implemented, validated and measured on RTX
5080. 500 GPU correctness tests passed, no skips; Compute Sanitizer memcheck, racecheck,
synccheck and initcheck: 0 errors, 0 hazards. Source: [LeetGPU Matrix
Transpose](https://leetgpu.com/challenges/matrix-transpose) (row-major float32, 1 ≤ rows,
cols ≤ 8192; performance case 7000 × 6000).

![Effective bandwidth against logical traffic per call: past the L2 boundary the tiled CUDA kernels and Triton reach about 745 GB/s (FP32) and 705–727 GB/s (FP16) against the plain-copy ceiling (memcpy, about 820; not a transpose); pytorch and the naive kernels stay far below.](figures/bandwidth_vs_size.svg)

## Key results

`memcpy` (`a.clone()`) and copy `cuda_vec` (07's pack kernel) are plain copies, not
transposes: they are the bandwidth ceiling a transpose can at best match, not competitors.

- **Transpose reaches 91–92% of the copy ceiling in FP32 and 88–89% in FP16** at the
  source's 7000 × 6000 (`triton` 450.1 and 231.1 µs, `cuda_vec_ilp` 453.0 and 233.1 µs;
  memcpy 412.1 and 205.3 µs), moving exactly the minimum DRAM bytes. Every tiled kernel is
  2.1–2.6× faster than PyTorch's `a.t().clone()`.
- **Bank conflicts measured exactly as modelled:** unpadded column reads cost 32.1 (FP32)
  and 16.1 (FP16) shared wavefronts per load; padding brings FP32 to 1.00 but leaves FP16
  at 1.50; the XOR swizzle is 1.00 in both.
- **Conflicts matter only off DRAM.** At 7000 × 6000 the unpadded tile is 0.2% (FP32) and
  4% (FP16) slower; L2-resident at 2048² FP32 it is 1.8× slower (27.9 vs 15.2 µs).
- **Strided stores are the worse naive side,** except FP32 from DRAM: FP16 `cuda_naive_row`
  runs at 17% of the memcpy ceiling, 2.4× slower than `cuda_naive_col`.
- **With data in L2, 16-byte packs win per call:** `cuda_vec_ilp` is at 90–97% of the copy
  `cuda_vec` ceiling and 1.2–1.5× faster than the 32×32 tiles.
- **Triton gets there without hand-written swizzles:** its compiler-generated layout
  conversion is conflict-free, it is the fastest transpose DRAM-bound (by 0.6–0.9%) and as
  fast as `cuda_vec_ilp` kernel-only in L2; its call costs about 4 µs more.

`cuda_vec_ilp` is the recommended CUDA kernel: level with the best when DRAM-bound and the
fastest per call in L2. `cuda_smem_pad` (the `cuda` alias) is the simplest kernel within about
2% of it from DRAM.

## Documentation

| Page | Contents |
|---|---|
| [Design](docs/design.md) | Source, contract, reference, performance model, variants and hypotheses |
| [Results](docs/results.md) | Three-run tables, bank conflicts, naive sides, the L2 cliff, Triton, SASS, answers and limitations |
| [Testing](docs/testing.md) | Bit-exact suite, misalignment, binding checks, sanitizers |
| [Benchmarking](docs/benchmarking.md) | Reproduce the sweep and what one sample measures |
| [Profiling](docs/profiling.md) | Nsight captures, report sets and how to read the evidence |

## Quick start

```bash
python -m pytest -q problems/matrix_transpose
PYTHON=python bash problems/matrix_transpose/benchmarks/sweep.sh
```

```python
import torch
from problems.matrix_transpose.api import transpose

a = torch.randn(7000, 6000, device="cuda")
b = transpose(a, backend="cuda_vec_ilp")  # a new (6000, 7000) tensor, b[j, i] == a[i, j]
```

Backends: `pytorch` (`a.t().clone()`), `cuda_naive_row`, `cuda_naive_col`, `cuda_smem`,
`cuda_smem_pad` (the `cuda` alias), `cuda_smem_swizzle`, `cuda_vec_ilp` and `triton`
(64 × 64 tiles). Triton takes `block_m`, `block_n` and `num_warps`. FP32, FP16 and BF16;
empty matrices and dimensions above 8192 are lab extensions.

```text
matrix_transpose/
    api.py         validated public contract
    cases.py       test and benchmark inputs, bytes and FLOPs
    pytorch/       materialized transpose reference
    cuda/          kernels.cu, bindings.cpp, implementation.py (six variants)
    triton/        64 × 64 tile kernel
    tests/         test_matrix_transpose.py
    benchmarks/    sweep, copy control, timeline driver, summary and plot
    docs/          design, results, testing, benchmarking, profiling
```

Raw benchmark JSON, 46 Nsight Compute reports, an Nsight Systems timeline, CSV exports,
sanitizer logs and disassembly are in `artifacts/matrix_transpose/`, in this workspace and
on cuda.0. Large generated files are ignored by git;
[compact evidence](benchmarks/measured_summary.json), the scripts and the figure are
tracked.

Prerequisites: [07 · Matrix Copy](../matrix_copy/README.md). Follow the
[optimization methodology](../../docs/methodology.md).
