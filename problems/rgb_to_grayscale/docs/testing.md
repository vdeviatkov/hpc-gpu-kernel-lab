# Testing

[← RGB to Grayscale](../README.md) · [Design](design.md) · [Results](results.md) ·
**Testing** · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

Setup (virtual environment, CUDA toolkit, extension builds) is shared with
[Vector Addition](../../vector_add/docs/testing.md#setup).

## Running

```bash
python -m pytest -q problems/rgb_to_grayscale
ruff check . && ruff format --check .
clang-format --dry-run --Werror problems/rgb_to_grayscale/cuda/*.{cpp,cu}
```

Without an NVIDIA GPU the CUDA and `pytorch_compile` tests skip (`torch.compile` needs a
C++ toolchain on the CPU and adds nothing to this problem's coverage there); the reference,
`pytorch_matmul` and contract tests still run. Triton tests run on NVIDIA hardware
when Triton is installed.

## What is tested

[test_rgb_to_grayscale.py](../tests/test_rgb_to_grayscale.py), with inputs from
[cases.py](../cases.py):

- **Source examples** on every backend, against the statement's published values with its
  1e−5 tolerance: the two worked examples, all zeros and all 255.
- **Channel order,** in every dtype: pure red, green and blue pixels must give
  `255 · 0.299`, `255 · 0.587` and `255 · 0.114`; a swapped channel gives a different
  coefficient.
- **Fresh output:** a contiguous `(H, W)` tensor of the image's dtype and device that
  shares no memory with the image.
- **Misaligned image:** a row slice of a 9 × 13 image starts 39 elements into its
  allocation, not 16-byte aligned, so `cuda_vec` and `cuda_smem` must take their fallback.
- **Every backend against the reference** for FP32, FP16 and BF16 on the source's image
  sizes, empty images, and sizes whose pixel counts cover every remainder mod 8 (the
  vector path's scalar tail), up to 127 × 129; integer values 0–255 and real values in
  [0, 255].
- **Contract:** 4 channels, 2-D input, non-interleaved (permuted) and strided images,
  FP64, autograd and non-tensor inputs are rejected, as are unknown backends.

`pytorch_compile` compiles 69 distinct shapes and dtypes in one test session, well within
its recompile limit of 256.

## Compute Sanitizer

```bash
compute-sanitizer --tool memcheck python -m pytest -q problems/rgb_to_grayscale -k cuda
for tool in racecheck synccheck initcheck; do
  compute-sanitizer --tool $tool python -m pytest -q problems/rgb_to_grayscale \
    -k "cuda_smem or cuda_vec"
done
```

`racecheck` and `synccheck` matter for `cuda_smem`, the only kernel with shared memory and
a barrier; `initcheck` catches reads of uninitialized memory, such as tile slots a partial
tile never loads. The latest outcome is in [results](results.md#validation).

## Triton validation (2026-10-23)

On cuda.0 (RTX 5080, Triton 3.8.0), the combined problem 4 and 5 suites passed
**1,373 tests with no skips**. This includes FP32/FP16/BF16, empty inputs,
misalignment/storage offsets, tails, and added tile/warp configurations
`(64,1)`, `(128,4)`, `(256,8)` and `(1024,4)`.

```bash
python -m pytest -q problems/interleave_arrays problems/rgb_to_grayscale
compute-sanitizer --tool memcheck --target-processes all --error-exitcode 99 \
  python -m pytest -q problems/interleave_arrays problems/rgb_to_grayscale -k triton
```

Memcheck: **227 passed, 1,146 deselected; zero errors**. Logs are in
`artifacts/triton_04_05_20261023/` locally and
`/tmp/hpc-triton-04-05-20261023/artifacts/` on cuda.0. The remote run used an
isolated copy so the existing remote checkout was not changed.
