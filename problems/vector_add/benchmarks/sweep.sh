#!/usr/bin/env bash
# Reproduce the recorded measurement matrix. Run from the repository root after the
# correctness and sanitizer checks pass. Usage: sweep.sh [output_dir] [peak_gb_s]
set -euo pipefail
out=${1:-artifacts/vector_add}
peak=${2:-}
python=${PYTHON:-python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PWD/artifacts/torch_extensions}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$out"
run() { "$python" -m problems.vector_add.benchmarks.run ${peak:+--peak-gb-s "$peak"} "$@"; }
cuda=(cuda_scalar cuda_grid_stride cuda_vec cuda_vec_grid_stride)

# Default matrix (all dtypes), repeated to expose run-to-run noise.
for i in 1 2 3; do
  run --output "$out/device_$i.json" --backends pytorch "${cuda[@]}" triton
done

# Size sweep across the launch-bound, L2-resident, and DRAM-bound regimes.
sizes=(1024 4096 16384 65536 262144 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728)
for dtype in fp32 fp16; do
  run --output "$out/sizes_$dtype.json" --dtypes "$dtype" --sizes "${sizes[@]}" \
    --backends pytorch "${cuda[@]}" triton
done

# Host-observed API scope, including JAX.
run --scope api --output "$out/api.json" --backends pytorch "${cuda[@]}" triton jax

# Misaligned buffers: vector variants must take their scalar fallback.
run --offset 1 --sizes 1025 1048576 25000000 --dtypes fp32 fp16 \
  --backends pytorch "${cuda[@]}" --output "$out/offset_1.json"
