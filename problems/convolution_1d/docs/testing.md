# Testing

[← 1D Convolution](../README.md) · [Design](design.md) · [Results](results.md) · **Testing** ·
[Benchmarking](benchmarking.md) · [Profiling](profiling.md)

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
TORCH_EXTENSIONS_DIR=$PWD/artifacts/torch_extensions \
.venv/bin/python -m pytest -q problems/convolution_1d
```

**622 passed, 60 skipped** on the RTX 5080 (cuda.0, 2026-11-14). The skips are expected: 27
are `cuda_unrolled` on K values it has no specialization for (it reports
`NotImplementedError`), and 33 are the JAX baseline, which is not implemented. Without an
NVIDIA GPU, the reference, oracle, tolerance and input-validation tests run (75); every GPU
test skips explicitly.

## How results are compared

Convolution does arithmetic, so a backend is not expected to match the reference bit for
bit. Two correct FP32-accumulating implementations differ by summation order and one final
rounding. The suite therefore checks three things, at increasing strictness (derivation and
measured margins in [design §2](design.md#2-correctness-reference)):

1. **The reference against an FP64 oracle**, within the rigorous componentwise bound for any
   FP32-accumulated dot product rounded once: `|ŷ − y| ≤ u_d·|y| + (1 + u_d)·γ_K·Σ|x·w| + η_d`.
2. **Every backend against the reference** with fixed `(rtol, atol)` per dtype, calibrated
   up to K = 2047 and shown to reject TF32 products and FP16/BF16 accumulation.
3. **Long filters (K ≥ 4096) against the same FP64 bound,** where FP32 summation noise
   outgrows the fixed tolerances.

## What the suite checks

All six CUDA variants, the `cuda` alias and `triton` run every test that takes a backend.

| Test | Covers |
|---|---|
| `test_reference_matches_oracle` | The PyTorch reference (cuDNN on the GPU, with TF32 switched on globally first) against fsum-of-FP64-products explicit loops, within the FP32-accumulation bound, on every `TEST_CASES` shape and dtype |
| `test_tolerances_cover_summation_order` | A sequential FP32 sum passes the fixed tolerances against the reference at K = 255, 1023, 2047 |
| `test_tolerances_reject_reduced_precision` | Negative controls: TF32 products (FP32) and FP16 accumulation fail the fixed tolerances at K = 255 |
| `test_reference_contract` | Shape N − K + 1, dtype, device, contiguity, a fresh buffer, unchanged inputs, storage offsets; small-integer data so equality is exact |
| `test_reference_is_correlation_not_convolution` | Filter orientation: no flip |
| `test_matches_reference` | Every backend on every `TEST_CASES` shape and dtype: K = N = 1, K = N = 5, 17/3, 1025/31, 300/255, 1100/1023, K = N = 2047, 2100/2047 |
| `test_misaligned_input` | `x`, `w` or both starting one element into their storage (not 16-byte aligned), every backend and dtype |
| `test_binding_misaligned_output` | A misaligned output through the extension directly (the API always allocates an aligned one), every CUDA variant and dtype; guard elements must stay untouched |
| `test_cuda_const_capacity` | `cuda_const` at exactly 16384 taps (the whole 64 KiB constant bank), and rejection at 16385; the Python and C++ limits must agree |
| `test_cuda_const_reloads_filter` | Back-to-back `cuda_const` calls with long, short, long and medium filters and no host synchronization: no stale filter in constant memory |
| `test_cuda_smem_chunk_boundaries` | `cuda_smem`, `cuda_smem_reg` and `cuda_large_k` at K = 1023, 1024, 1025, 2049, 3000 (around 1024-tap chunk boundaries) with a partial last block, every dtype |
| `test_cuda_smem_reg_window_boundaries` | `cuda_smem_reg` at K = 1, 2, 7, 8, 9, 15, 16, 17, 1031–1033 (register-window group tails, in the first and second chunk) against 1, 2047, 2048, 2049 and 5000 outputs (block and per-thread boundaries) |
| `test_cuda_reg_outputs` | `cuda_smem_reg` and `cuda_large_k` at every compiled R = 2, 4, 8, 16, with tap tails shorter than, equal to and longer than R and a block-straddling output count |
| `test_cuda_unrolled` | Each specialization K = 3, 7, 31 at 1, 2047, 2048, 2049, 5000 outputs, `x` aligned and offset, every dtype |
| `test_cuda_unrolled_other_k_not_implemented` | Other K raise `NotImplementedError` through the API and `RuntimeError` through the binding |
| `test_cuda_large_k` | `cuda_naive`, `cuda_smem`, `cuda_smem_reg`, `cuda_large_k` at K = 4096, 16385, 20000, 65537 (far past a block, a chunk and the constant-memory limit), every dtype, against the FP64 bound |
| `test_triton_tile_shapes`, `test_triton_rejects_invalid_options` | Programs smaller than a warp, a single partial program, many full ones; invalid `block_size` / `num_warps` |
| `test_binding_rejects_invalid_arguments`, `test_cuda_rejects_invalid_*`, `test_rejects_*`, `test_gpu_backends_reject_cpu` | Variant and `reg_outputs` out of range, wrong output length, dtype mismatch, FP64, host tensors, 2-D and non-contiguous inputs, K > N, K = 0, overlapping buffers, autograd, unknown backend |
| `test_cuda_alias_is_smem_reg` | `cuda` resolves to `cuda_smem_reg` |
| `test_traffic_model` | `logical_bytes` = (2N + 1)·e and `flops` = 2·K·(N − K + 1) |

## Compute Sanitizer

```bash
for tool in memcheck racecheck synccheck initcheck; do
  compute-sanitizer --tool $tool --target-processes all --error-exitcode 1 \
    .venv/bin/python -m pytest -q problems/convolution_1d -k "cuda or triton"
done
```

| Tool | Result |
|---|---|
| memcheck | 549 passed, 27 skipped; ERROR SUMMARY: 0 errors |
| racecheck | 549 passed, 27 skipped; RACECHECK SUMMARY: 0 hazards (0 errors, 0 warnings) |
| synccheck | 549 passed, 27 skipped; ERROR SUMMARY: 0 errors |
| initcheck | 549 passed, 27 skipped; ERROR SUMMARY: 0 errors |

This covers every CUDA variant, including the chunk loops, partial last blocks whose idle
threads must still reach every barrier, the register-window tails and the constant-memory
upload, and the Triton kernel. Logs: `artifacts/convolution_1d/sanitizer_*.log`;
[sweep.sh](../benchmarks/sweep.sh) runs the same four tools before any timing and stops on a
failure.

One earlier memcheck run (2026-11-03, Triton only, before the CUDA variants existed) reported
a single error that did not recur in 13 reruns, 8 of them with a fresh Triton cache; its log
was not kept, so its source is unknown. Every run since has been clean.
