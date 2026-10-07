# Testing

[← ReLU](../README.md) · [Design](design.md) · [Results](results.md) ·
**Testing** · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

Setup (virtual environment, CUDA toolkit, extension builds) is shared with
[Vector Addition](../../vector_add/docs/testing.md#setup).

## Running

```bash
python -m pytest -q problems/relu
ruff check . && ruff format --check .
clang-format --dry-run --Werror problems/relu/cuda/*.{cpp,cu}
```

Without an NVIDIA GPU the CUDA and Triton tests skip and only the reference and contract
tests run.

## What is tested

[test_relu.py](../tests/test_relu.py), with inputs from [cases.py](../cases.py):

- **Source examples:** the two worked examples from the LeetGPU statement, on every
  backend including the reference.
- **Every backend against `torch.relu`, bit for bit,** for FP32, FP16 and BF16 on:
  - sizes 0, 1, 2, 3, 5, 1024, 1025, 10000, 65537 with mixed signs (empty input, the
    source's sizes, and tails that do not fill a 16-byte pack);
  - N = 1025 and 10000 for each distribution: positive, negative, alternating signs,
    warp-sized sign blocks, zeros, ±1000 and ±0.001 (the source's value ranges).
- **Inputs:** the alternating and warp-block inputs really have the intended pattern.
- **Contract:** non-tensors, 2-D, strided and FP64 inputs are rejected, as are unknown
  backends.

Inputs are cloned per backend, and each backend must leave its input unchanged.
Special values (NaN, ±inf, −0.0) are not part of the suite.

## Compute Sanitizer

```bash
compute-sanitizer --tool memcheck python -m pytest -q problems/relu -k "cuda or triton"
```

The latest outcome is in [results](results.md#validation).
