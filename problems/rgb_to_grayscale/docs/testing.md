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
`pytorch_matmul` and contract tests still run. The Triton tests skip as "not implemented".

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
