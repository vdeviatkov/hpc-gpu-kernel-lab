# Testing

[← Matrix Copy](../README.md) · [Design](design.md) · [Results](results.md) · **Testing** ·
[Benchmarking](benchmarking.md) · [Profiling](profiling.md)

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
TORCH_EXTENSIONS_DIR=$PWD/artifacts/torch_extensions \
.venv/bin/python -m pytest -q problems/matrix_copy
```

**1436 passed, no skips** on the RTX 5080 (cuda.0). Without an NVIDIA GPU, only the
PyTorch reference and input validation run (17 tests); the rest skip explicitly.

## What the suite checks

Every backend is compared bit for bit: outputs are viewed as int32/int16 and compared
exactly, because `assert_close` treats −0.0 and +0.0 as equal and any two NaNs as equal.

| Test | Covers |
|---|---|
| `test_source_examples` | The statement's two worked examples |
| `test_source_functional_inputs` | The source checker's five other inputs: zeros, identity, negatives, uniform 16 × 16, a single element |
| `test_special_values_bit_exact` | ±0, ±inf, quiet and signalling NaNs with distinct payloads and signs, smallest and largest subnormals, largest finite, 1.0, at every row and column of a 7 × 13 matrix next to random bit patterns; every dtype |
| `test_fresh_output` | A new contiguous tensor of the same shape, dtype and device; writing it leaves the input unchanged |
| `test_misaligned_input` | A matrix starting one element into its storage (not 16-byte aligned): vector paths must fall back |
| `test_matches_reference` | Every `TEST_CASES` shape in every dtype: N = 1…1025 covering every N mod 4 and mod 8, warp and block boundaries, rectangular shapes with tails, 600000 × 1 and 1 × 600000 (beyond 65535 grid-y blocks), and empty matrices |
| `test_cuda_block_sizes` | Every CUDA variant at 128 and 512 threads |
| `test_binding_misaligned_buffers` | Misaligned input, output, or both, through the extension directly (the API always allocates an aligned output); a guard element checks nothing is written outside the output |
| `test_binding_rejects_invalid_arguments` | Variant, block size, 1-D, non-contiguous, shape and dtype mismatch, FP64, host tensors, output equal to or partially overlapping the input |
| `test_triton_tile_shapes` | Non-square tiles: 16 × 128, 128 × 16, 32 × 32, 1 × 1024 |
| `test_rejects_*` | Invalid API inputs, unknown backend, unknown PyTorch method, invalid Triton options |

The bit-exact test was checked against a deliberately wrong backend that canonicalizes NaN
and adds +0.0: it fails in all three dtypes.

## Compute Sanitizer

```bash
compute-sanitizer --tool memcheck --target-processes all --error-exitcode 99 \
  .venv/bin/python -m pytest -q problems/matrix_copy \
  -k "cuda and (misaligned or matches_reference or block_sizes)"
```

Result: **1076 passed, 360 deselected; ERROR SUMMARY: 0 errors**, exit 0. This covers the
CUDA variants, including the misaligned fallbacks; it makes no claim about Triton or the
PyTorch reference.
