# Testing

[← Interleave Arrays](../README.md) · [Design](design.md) · [Results](results.md) ·
**Testing** · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

Setup (virtual environment, CUDA toolkit, extension builds) is shared with
[Vector Addition](../../vector_add/docs/testing.md#setup).

## Running

```bash
python -m pytest -q problems/interleave_arrays
ruff check . && ruff format --check .
clang-format --dry-run --Werror problems/interleave_arrays/cuda/*.{cpp,cu}
```

Without an NVIDIA GPU the CUDA tests skip; the reference, `pytorch_stack` and contract
tests still run. The Triton tests always skip ("not implemented"): Triton is deferred.

## What is tested

[test_interleave_arrays.py](../tests/test_interleave_arrays.py), with inputs from
[cases.py](../cases.py):

- **Source examples:** the statement's two worked examples plus inputs in the spirit of its
  functional cases (single element, negatives, mixed signs, zeros and ones, ±1e10 and
  ±1e−10), on every backend. Expected outputs come from a plain-Python interleave, so the
  reference is checked too.
- **Output order,** in every dtype: `a = 0, 2, 4, …` and `b = 1, 3, 5, …` must interleave to
  exactly `0, 1, 2, …`, which exposes any swapped, shifted or duplicated element.
- **Fresh output:** 2N elements, input dtype and device, contiguous, sharing no memory with
  either input; and the same tensor passed as both inputs.
- **Storage offsets:** inputs starting 1 or 3 elements into an allocation, so they are not
  16-byte aligned and `cuda_vec` must use its fallback.
- **Every backend against the reference, bit for bit,** for FP32, FP16 and BF16 on N = 0–9,
  15–17, 31–33, 255–257, 1024, 1025, 10000, 65537 and 100000 (every N mod 8, warp and block
  boundaries), with inputs required to come back unchanged.
- **Contract:** mismatched lengths or dtypes, 2-D, strided, FP64, autograd and non-tensor
  inputs are rejected, as are unknown backends.

## Compute Sanitizer

```bash
compute-sanitizer --tool memcheck python -m pytest -q problems/interleave_arrays -k cuda
```

The latest outcome is in [results](results.md#validation).
