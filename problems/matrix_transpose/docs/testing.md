# Testing

[← Matrix Transpose](../README.md) · [Design](design.md) · [Results](results.md) · **Testing** ·
[Benchmarking](benchmarking.md) · [Profiling](profiling.md)

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
TORCH_EXTENSIONS_DIR=$PWD/artifacts/torch_extensions \
.venv/bin/python -m pytest -q problems/matrix_transpose
```

**500 passed, no skips** on the RTX 5080 (cuda.0, 2026-11-02). Without an NVIDIA GPU, only the
PyTorch reference and input validation run (56 tests); the rest skip explicitly.

## What the suite checks

Every GPU backend is compared bit for bit with a transposed integer view of its input,
because `assert_close` treats −0.0 and +0.0 as equal and any two NaNs as equal. All six CUDA
variants, the `cuda` alias and `triton` run every test that takes a backend.

| Test | Covers |
|---|---|
| `test_reference_contract` | The PyTorch reference against an independently built transpose (indexed assignment), on every `TEST_CASES` shape and dtype: output shape (N, M), dtype, device, contiguity, a fresh buffer, an unchanged input, and a storage offset |
| `test_reference_preserves_bits` | The reference on every 16-bit pattern (FP16, BF16) and on random FP32 patterns plus ±0, ±inf and distinct NaN payloads |
| `test_special_values_bit_exact` | ±0, ±inf, quiet and signalling NaNs with distinct payloads and signs, smallest and largest subnormals, largest finite, 1.0, at every row and column of a 7 × 13 matrix next to random bit patterns; every backend and dtype |
| `test_matches_reference` | Every `TEST_CASES` shape in every dtype: 1 × 1, 2 × 3 and 3 × 1 (the source's worked shapes), rows and columns of one, warp and tile boundaries (31 × 32, 32 × 33, 65 × 129, 129 × 65), 40 × 520 (full 16 × 256 `cuda_vec_ilp` tiles plus tails), 8193 × 1, and empty 0 × 0, 0 × 5, 5 × 0 |
| `test_misaligned_input` | A 40 × 520 matrix starting one element into its storage (not 16-byte aligned) through the API: `cuda_vec_ilp` must take its scalar path; every backend and dtype |
| `test_binding_misaligned_buffers` | Misaligned input, output, or both, through the extension directly (the API always allocates an aligned output), for every CUDA variant and dtype; a guard element checks nothing is written outside the output |
| `test_binding_rejects_invalid_arguments` | Variant out of range, 1-D, non-contiguous input, shape not reversed, dtype mismatch, FP64, host tensors, partially overlapping input and output |
| `test_triton_tile_shapes` | Non-square Triton tiles: 16 × 128, 128 × 16, 32 × 32, 1 × 1024 |
| `test_triton_rejects_invalid_options` | `block_m` = 0, `block_n` = 48 (not a power of two), `num_warps` = 3 |
| `test_gpu_backends_reject_cpu`, `test_rejects_*` | Host tensors on GPU backends; invalid API inputs (1-D, 3-D, transposed and strided views, FP64, int32, sparse, non-tensors); unknown backend |
| `test_traffic_model` | `logical_bytes` = 2·M·N·e for the source case |

The bit-exact test was checked against a deliberately wrong backend that transposes through
arithmetic (`a.t() + 0.0`, which canonicalizes NaNs and turns −0.0 into +0.0): it fails in
all three dtypes.

## Compute Sanitizer

```bash
for tool in memcheck racecheck synccheck initcheck; do
  compute-sanitizer --tool $tool --target-processes all --error-exitcode 1 \
    .venv/bin/python -m pytest -q problems/matrix_transpose -k "cuda or triton"
done
```

| Tool | Result |
|---|---|
| memcheck | 445 passed, 55 deselected; ERROR SUMMARY: 0 errors |
| racecheck | 445 passed; RACECHECK SUMMARY: 0 hazards (0 errors, 0 warnings) |
| synccheck | 445 passed; ERROR SUMMARY: 0 errors |
| initcheck | 445 passed; ERROR SUMMARY: 0 errors |

This covers every CUDA variant, including the misaligned and edge-tile scalar paths and the
shared-memory barriers, and the Triton kernel. It makes no claim about the PyTorch reference.
Logs: `artifacts/matrix_transpose/sanitizer_*.log`; [sweep.sh](../benchmarks/sweep.sh) runs
the same four tools before any timing and stops on a failure.
