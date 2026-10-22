# Testing

[← Rainbow Table](../README.md) · [Results](results.md)

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
TORCH_EXTENSIONS_DIR=$PWD/artifacts/torch_extensions \
.venv/bin/python -m pytest -q problems/rainbow_table
```

All 510 tests passed on the remote RTX 5080. The suite checks published source
examples, an independent Python integer oracle, signed edge bit patterns, random
full-range int32 values, R = 0/1/2/7/101, fresh output and input preservation,
misaligned vector input, empty input, tails and warp/block boundaries. Every
implemented backend is tested bit for bit. GPU execution covers Triton looped and
unrolled kernels as well as compiled PyTorch. A CPU-only pass cannot validate them.

For a focused memory-safety check after correctness, use Compute Sanitizer's
memcheck on the misalignment and reference-comparison tests for CUDA backends.
Record the command, selected tests, exit status and log; passing ordinary
correctness tests is not a sanitizer result.

Recorded memcheck command on cuda.0:

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
TORCH_EXTENSIONS_DIR=$PWD/artifacts/torch_extensions \
compute-sanitizer --tool memcheck --target-processes all --error-exitcode 99 \
  .venv/bin/python -m pytest -q problems/rainbow_table \
  -k "(misaligned_input or matches_reference) and cuda"
```

Result: **238 passed, 272 deselected; ERROR SUMMARY: 0 errors**, exit 0.
The log is `artifacts/rainbow_table/memcheck.log`. This focused check covers CUDA
variants; it does not claim sanitizer coverage for Triton or eager PyTorch.
