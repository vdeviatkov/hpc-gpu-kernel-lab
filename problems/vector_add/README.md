# 01 · Vector Addition

A first streaming-kernel study: separate launch costs from memory throughput,
then test what vectorized access and grid-stride loops each change.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/bandwidth_vs_size_dark.svg">
  <img alt="Effective bandwidth vs working-set size on RTX 5080 for PyTorch, CUDA scalar, CUDA vector and Triton, FP32 and FP16. Bandwidth rises linearly with size while launch-bound, peaks near 3 TB/s while the working set fits in the 64 MiB L2, and settles near 840 GB/s once it streams from DRAM." src="figures/bandwidth_vs_size_light.svg">
</picture>

## Key results

Measured on an RTX 5080 (spec peak 960 GB/s). Details and caveats are in [Results](#results).

- **Three regimes.** Below ~1 MB, each call costs ~8.3 µs regardless of size
  (launch-bound; Triton ~13.7 µs). While A + B + C fits in the 64 MiB L2, effective
  bandwidth reaches ~3 TB/s. Beyond it, every backend streams from DRAM at
  816–863 GB/s at N = 25M, which is 85–90% of the spec peak.
- **16-byte vector access matters for 2-byte types.** At N = 25M FP16, `cuda_vec`
  takes 173.7 µs vs 183.8 µs for `cuda_scalar` (−5.5%) and 177.6 µs for PyTorch.
  Instructions fall 72% and DRAM throughput rises from 88.5% to 92.6%.
- **For FP32, the grid-stride loop matters more than vectorization.** Vectorization changes
  latency by about ±1%, while adding the capped grid-stride loop costs up to 2.6%: more registers
  (34–54 vs 16–18) and, for the vector loop, fewer active warps (62% vs 86%).
  Plain `cuda_scalar` is the fastest FP32 variant (354.8 µs, 88% of peak).

## Source and contract

[LeetGPU Vector Addition](https://leetgpu.com/challenges/vector-addition), Easy.
Inspected on 2026-09-12: equal-length FP32 vectors, `C[i] = A[i] + B[i]`,
1 ≤ N ≤ 100,000,000; the performance case uses N = 25,000,000.
These are library-integrated experiments, not challenge submissions.

**Lab extensions:** empty vectors, FP16/BF16, and contiguous storage-offset views.
The checked entry point is `api.add(a, b, backend=..., out=None)`:

- Equal contiguous 1-D tensors, matching dtype/device; no broadcasting.
- FP32 arithmetic, with FP16/BF16 outputs rounded after an FP32 intermediate.
- Inputs may alias each other; output must not overlap either input. Disjoint
  slices of the same storage are allowed. Inputs are never mutated.
- Fresh output unless `out` is supplied; empty inputs launch no device kernel.
- Forward-only: autograd and unresolved negative/conjugate views are rejected.
- CUDA/Triton use the input device and PyTorch's current stream, asynchronously.
  Consumers on other streams must establish their own dependencies.
- CPU PyTorch/JAX support correctness checks, not GPU performance claims.

No prerequisites. Primary lesson: **launch overhead vs bandwidth**.

## Implementations

The four CUDA variants form a 2×2 design, so each comparison changes one factor:

| | One item per thread | Capped grid-stride loop (≤ 4096 blocks) |
|---|---|---|
| **Scalar access** | `cuda_scalar` | `cuda_grid_stride` |
| **16-byte vector access** | `cuda_vec` | `cuda_vec_grid_stride` |

A 16-byte pack holds four FP32 or eight FP16/BF16 elements, loaded with
`LDG.E.128` and stored with `STG.E.128` (verified in SASS for all three dtypes). Elements that
do not fill a whole pack are handled in the same launch. If any of the three pointers is not
16-byte aligned, the vector variant falls back along the vector axis only (`cuda_vec` →
`cuda_scalar`, `cuda_vec_grid_stride` → `cuda_grid_stride`), so the loop structure is kept.
All pointers are `__restrict__`; inputs therefore use the read-only `.CONSTANT` load path.

| Other backends | Mechanism |
|---|---|
| `pytorch` | Native eager add, with optional output buffer |
| `triton` | Masked tiles; default 1024 elements and 4 warps |
| JAX | Native arrays with JIT compilation and explicit completion |

Read [CUDA kernels](cuda/kernels.cu), [bindings](cuda/bindings.cpp),
[Triton](triton/implementation.py), [PyTorch](pytorch/implementation.py),
and [JAX](jax/implementation.py). All support FP32/FP16/BF16.

[api.py](api.py) owns the public PyTorch-tensor contract; backend modules are
internal. JAX exposes a separate native-array interface. CUDA repeats validation
at the C++ boundary, uses an RAII device guard and checks launch errors without
inserting synchronization.

## Performance model

N additions move a minimum of `3*N*s` logical bytes for element size s: two
reads and one write. Arithmetic intensity is `1/(3*s)` FLOP/byte, or 1/12
for FP32. At N = 25,000,000, logical traffic is 300 MB for FP32.

At a bandwidth ceiling W bytes/s, `3*N*s/W` is an optimistic memory-time bound.
The RTX 5080's spec ceiling is 960 GB/s (16 GB GDDR7, 256-bit bus at 30 Gbps), so
25M FP32 elements need at least 312.5 µs. Effective GB/s is `3*N*s / seconds / 1e9`,
not measured DRAM traffic. There is no data reuse to justify shared memory and no
Tensor Core operation here.

Every benchmark case also times a device-to-device `Tensor.copy_` (`copy_reference`)
as a measured bandwidth reference. Copy is half writes, while addition is one third
writes, so copy is **not** a strict upper bound: for FP32 at 25M elements, addition
(846 GB/s) beats copy (817 GB/s). The spec peak is the ceiling; the copy reference
shows what a vendor-tuned streaming operation achieves on the same buffers.

## Results

### Measurement environment

Run date: 2026-10-02 UTC. RTX 5080 (SM120, 84 SMs, 64 MiB L2), driver 595.91.07,
CUDA toolkit 13.2.86, Python 3.12.3, PyTorch 2.14.1 (CUDA 13.0), Triton 3.8.0,
JAX/jaxlib 0.11.2. Stock clocks, desktop display active, one GPU; clocks not locked.
Each case: correctness check, 25 warmups, 100 samples. Buffers are reused without
cache flushing. Backend order is shuffled per case with a fixed seed.
Measured from the working tree of this revision (base commit `3001d91` plus the
uncommitted changes recorded in the JSON metadata).

### N = 25,000,000 (DRAM-bound)

Median of three complete run medians; the range shows those three medians, not a
confidence interval. Scope: CUDA events around an eager call on the current stream.

| Backend | FP32 µs | FP32 GB/s | FP16 µs | FP16 GB/s | BF16 µs | BF16 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| `pytorch` | 359.26 (358.86–359.63) | 835.0 | 177.60 (177.57–177.66) | 844.6 | **175.55** (175.46–175.65) | 854.4 |
| `cuda_scalar` | **354.78** (354.59–355.60) | 845.6 | 183.81 (183.81–183.84) | 816.1 | 183.81 (183.81–183.87) | 816.1 |
| `cuda_grid_stride` | 364.03 (363.74–364.06) | 824.1 | 182.78 (182.75–182.78) | 820.6 | 180.77 (180.74–180.83) | 829.8 |
| `cuda_vec` | 358.64 (358.00–358.88) | 836.5 | **173.73** (173.63–174.32) | 863.4 | 175.68 (175.63–175.71) | 853.8 |
| `cuda_vec_grid_stride` | 359.78 (359.76–359.81) | 833.9 | 178.72 (178.66–178.75) | 839.3 | 179.49 (179.49–179.55) | 835.7 |
| `triton` | 363.95 (363.90–364.02) | 824.3 | 179.41 (179.39–179.55) | 836.1 | 181.84 (181.82–182.45) | 824.9 |
| `copy_reference` | 244.70 | 817.3 | 116.08 | 861.5 | 114.18 | 875.8 |

`copy_reference` GB/s counts two streams (`2*N*s`); all other rows count three.

### What the 2×2 design shows

Nsight Compute, one profiled launch per variant at N = 25M. Profiler replay controls caches
and clocks differently from normal timing, so durations differ from the table above. The
ordering matches.

| Variant | Dtype | Duration µs | DRAM % of peak | Warps active % | Instructions | Registers |
|---|---|---:|---:|---:|---:|---:|
| `cuda_scalar` | FP32 | 302.7 | 91.8 | 82.6 | 17.19 M | 16 |
| `cuda_vec` | FP32 | 304.2 | 91.3 | 86.5 | 6.25 M | 18 |
| `cuda_grid_stride` | FP32 | 322.5 | 89.8 | 92.1 | 9.31 M | 34 |
| `cuda_vec_grid_stride` | FP32 | 313.0 | 90.7 | 61.9 | 6.14 M | 54 |
| `cuda_scalar` | FP16 | 156.4 | 88.5 | 80.7 | 19.53 M | 16 |
| `cuda_vec` | FP16 | 128.1 | 92.6 | 87.2 | 5.47 M | 26 |
| `cuda_grid_stride` | FP16 | 140.8 | 91.5 | 92.8 | 11.85 M | 29 |
| `cuda_vec_grid_stride` | FP16 | 128.4 | 93.5 | 76.0 | 6.79 M | 48 |

- **Vectorization, loop held fixed.** FP32: scalar → vec cuts instructions 64% but
  leaves DRAM throughput unchanged; latency moves +1.1% (direct) and −1.2% (grid-stride).
  At four bytes per element, scalar loads already keep DRAM about 92% busy. FP16: the
  same change cuts instructions 72% and is 5.5% faster (direct). Two-byte scalar accesses
  issue too many memory instructions to keep DRAM busy.
- **Grid-stride loop, access width held fixed.** FP32 scalar → grid-stride is 2.6%
  slower. The compiler unrolls the loop (34 registers vs 16) and the 4096-block cap is a
  fixed starting choice. The vector loop reaches 54 registers and 62% active warps.
  An earlier README compared the vector loop against `cuda_scalar` and attributed the gap
  to vectorization; the 2×2 design shows it came from the loop.
- **Best per dtype.** FP32: `cuda_scalar`, 1.3% faster than PyTorch. FP16: `cuda_vec`,
  2.2% faster than PyTorch. BF16: `cuda_vec` and PyTorch tie (within 0.1%).

### Size sweep and small vectors

The chart above comes from 13 sizes per dtype ([plot script](benchmarks/plot.py)).
At N = 1025, device-scope medians are 8.3–8.5 µs for PyTorch and all CUDA variants,
13.7 µs for Triton, and 5.4 µs for `copy_reference`. An earlier Nsight Systems capture
of the scalar kernel at N = 1025 measured 0.61 µs median kernel duration and 1.92 µs
median `cudaLaunchKernel` time, so event timing at this size is mostly
dispatch, not memory traffic. The L2 peak is at 50 MB for FP32 (N = 4,194,304; 3.1 TB/s); at 101 MB,
the working set no longer fits and bandwidth drops to ~1 TB/s, then to the DRAM plateau.

### API scope (one run, FP32)

Host-observed allocation, dispatch and completion; JAX receives identical GPU inputs
through DLPack. Do not mix with the device-scope tables.

| Backend | N = 1025 µs | N = 25,000,000 µs |
|---|---:|---:|
| `pytorch` | 8.24 | 360.05 |
| `cuda_scalar` | 9.53 | 355.97 |
| `cuda_grid_stride` | 8.50 | 366.02 |
| `cuda_vec` | 8.52 | 359.46 |
| `cuda_vec_grid_stride` | 8.72 | 363.19 |
| `triton` | 13.58 | 366.00 |
| `jax` | 23.61 | 385.64 |

### Validation

- GPU test suite: **811 passed, 0 skipped**.
- Compute Sanitizer memcheck: **508 CUDA tests, 0 errors**. Covers values/tails, guard
  buffers, output alignment for every dtype, multi-iteration grid-stride loops and
  nondefault streams.
- A misaligned run (`--offset 1`) confirms that the vector variants take their scalar fallback
  and match its timing.
- CI runs lint, formatting and the CPU test suite on every push.

### Limitations

One desktop GPU, unlocked clocks, reused buffers, and one API-scope run. Multi-GPU
device selection is implemented (device guard) but not tested. Geometry sweeps
(threads, Triton tiles) were last run on the previous revision and showed effects
of about 1%; they are not repeated here. No universal fastest backend is claimed.
Useful follow-ups: rotating buffers around the L2 transition, an uncapped or
SM-count-derived grid for the grid-stride variants, and CUDA Graphs for small sizes.

## Setup and usage

From the repository root, use Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r problems/vector_add/requirements.txt
```

On Linux NVIDIA hosts, use a CUDA-enabled PyTorch wheel, matching driver/toolkit
and supported C++ compiler. Make nvcc discoverable through CUDA_HOME or PATH.
Install Triton separately if PyTorch does not supply it. JAX is optional; install
the CUDA extra appropriate to your setup. See the
[PyTorch selector](https://pytorch.org/get-started/locally/) and
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

The [loader](cuda/implementation.py) builds lazily with Ninja via
[PyTorch's extension API](https://docs.pytorch.org/docs/stable/cpp_extension.html).
It inherits PyTorch's required C++ standard and ABI, uses O3/line information,
and does not enable fast math. Set `TORCH_EXTENSIONS_DIR=artifacts/torch_extensions`
to keep the build cache here. Record any explicit `TORCH_CUDA_ARCH_LIST`.

```python
import torch

from problems.vector_add.api import add

a = torch.randn(1025, device="cuda")
b = torch.randn_like(a)
c = add(a, b, backend="cuda_vec")
torch.cuda.synchronize()
```

## Correctness tests

```bash
python -m pytest -q
ruff check problems
ruff format --check problems
clang-format --dry-run --Werror problems/vector_add/cuda/*.{cpp,cu}
```

Tests cover empty/odd/large vectors, partial-pack tails, independent output alignment
for every dtype, guard values, input preservation, cancellation and NaN/Inf, invalid
contracts, launch configurations and nondefault streams.
The reference adds FP32-converted inputs before rounding to the output dtype.
Tolerances are explicit in api.py; guard/ownership checks are exact.
Missing hardware/dependencies are explicit skips; build and correctness failures
fail. Inspect skips on the target host before acceptance.

## Benchmarking

```bash
python -m problems.vector_add.benchmarks.run --peak-gb-s 960 --output artifacts/vector_add.json
python -m problems.vector_add.benchmarks.report artifacts/vector_add.json

# Full recorded matrix: 3 default runs, size sweeps, API scope with JAX, misalignment.
PYTHON=.venv/bin/python TORCH_CUDA_ARCH_LIST=12.0 \
  problems/vector_add/benchmarks/sweep.sh artifacts/vector_add 960
python -m problems.vector_add.benchmarks.plot \
  artifacts/vector_add/sizes_fp32.json artifacts/vector_add/sizes_fp16.json
```

Each backend passes correctness before warmup/timing. Default sizes are 1, 1025,
1,048,576 and 25,000,000; dtypes are FP32/FP16/BF16. BF16 benchmarks require SM80+
by policy. Requested missing packages and build/runtime errors abort the run;
select available implementations with `--backends`. `--peak-gb-s` records a
sourced hardware ceiling and adds `fraction_of_peak`; without it, no ceiling is guessed.

- **device:** preallocated output, CUDA events on the current stream, synchronized
  after each sample. This measures stream elapsed time around an eager call;
  host submission gaps can enter it. It is not isolated kernel duration.
- **api:** host-observed allocation, dispatch and completion. PyTorch/CUDA/Triton
  synchronize the selected device; JAX waits on its result. Transfers and JIT/build
  are excluded. Tiny comparisons include runtime-specific completion costs.
  JAX receives the exact GPU inputs through DLPack, never through a CPU fallback.

Never mix these scopes in a speedup claim. Defaults are 25 warmups and 100 samples,
with raw microseconds, p20/p50/p80, mean/stddev, effective GB/s, elements/s,
fraction of the copy reference and matching-case speedup against PyTorch in JSON.
Metadata includes GPU, driver/toolkit/runtime, installed packages, source revision
and dirty state, configuration and `--notes` for clocks/load/thermal conditions.

## Profiling

Build/warm up before a CUDA profiler capture range. Choose one backend, size and
dtype; profile runs produce no benchmark JSON.

```bash
mkdir -p artifacts
ncu --profile-from-start off --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section LaunchStats --launch-count 1 -o artifacts/vector_add_ncu \
  python -m problems.vector_add.benchmarks.run --profile --backends cuda_vec \
  --sizes 25000000 --dtypes fp16 --samples 1

nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o artifacts/vector_add_nsys \
  python -m problems.vector_add.benchmarks.run --profile --backends cuda_scalar \
  --sizes 1025 --dtypes fp32 --samples 100

compute-sanitizer --tool memcheck python -m pytest -q \
  problems/vector_add/tests/test_vector_add.py -k 'cuda and (values_and_guards or alignment)'

# Registers and load/store widths of the built extension:
cuobjdump --dump-resource-usage artifacts/torch_extensions/gpu_lab_vector_add/*.so
cuobjdump -sass artifacts/torch_extensions/gpu_lab_vector_add/*.so | grep -E 'LDG|STG'
```

Raw JSON, Nsight Compute metrics and resource usage are kept in
`results/rtx5080/current/` in the local workspace. That directory is ignored by Git;
the tables above are the tracked summary.

## References

- [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Triton programming model](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [JAX asynchronous benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)
- [Lab methodology](../../docs/methodology.md) and [benchmark policy](../../docs/benchmarking.md)
