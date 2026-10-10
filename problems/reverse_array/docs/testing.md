# Testing

[← Reverse Array](../README.md) · [Design](design.md) · [Results](results.md) ·
**Testing** · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

Setup (virtual environment, CUDA toolkit, extension builds) is shared with
[Vector Addition](../../vector_add/docs/testing.md#setup).

## Running

```bash
python -m pytest -q problems/reverse_array
ruff check . && ruff format --check .
clang-format --dry-run --Werror problems/reverse_array/cuda/*.{cpp,cu}
```

Without an NVIDIA GPU the CUDA and Triton tests skip and only the reference and contract
tests run.

## What is tested

[test_reverse_array.py](../tests/test_reverse_array.py), with inputs from
[cases.py](../cases.py):

- **Source examples:** the statement's worked examples and explicit functional cases
  (single element, negatives, mixed signs, tiny and large values, 30 sequential values),
  plus an odd length that pins the middle element, on every backend.
- **In place:** the result is the input tensor itself, with the same storage address,
  now reversed.
- **Every backend against the reference, bit for bit,** for FP32, FP16 and BF16 on:
  - N = 0–9, 15–17, 30–33, 63–65, 511–514 (around warps and the 256-pair block
    boundary), 1000, 1023–1025, 10000, 65535–65537: odd and even, every N mod 8;
  - N = 4, 1000 and 10000 for each source value range (zeros, [0, 1), ±10⁻⁶, ±10⁷,
    clamped to the dtype's finite range).
- **Contract:** non-tensors, 2-D, strided, FP64 and autograd inputs are rejected, as are
  unknown backends.

## Compute Sanitizer

```bash
compute-sanitizer --tool memcheck python -m pytest -q problems/reverse_array -k "cuda or triton"
compute-sanitizer --tool racecheck python -m pytest -q problems/reverse_array -k "tile or triton"
compute-sanitizer --tool synccheck python -m pytest -q problems/reverse_array -k "tile_async"
```

racecheck matters here because the tile kernels and Triton exchange data through shared
memory. The latest outcome is in [results](results.md#validation).
