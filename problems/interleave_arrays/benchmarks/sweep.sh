#!/usr/bin/env bash
# Reproduce the recorded Interleave Arrays results. Run from the repository root on an
# NVIDIA host. Usage: sweep.sh [output_dir] [peak_gb_s]
set -euo pipefail
out=${1:-artifacts/interleave_arrays}
peak=${2:-960}
python=${PYTHON:-python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PWD/artifacts/torch_extensions}
mkdir -p "$out/ncu"
backends=(pytorch pytorch_stack cuda_input cuda_output cuda_pair_store cuda_vec triton)

# Correctness gate: never time or profile a backend that fails its tests.
"$python" -m pytest -q problems/interleave_arrays

# Three complete benchmark runs, with a copy of both inputs as the reference.
for i in 1 2 3; do
  "$python" -m lab.bench interleave_arrays --backends "${backends[@]}" --copy-reference \
    --peak-gb-s "$peak" --output "$out/device_$i.json"
done

# Full Nsight Compute reports (all sections plus per-instruction source counters), one
# launch per backend at the source's size, N = 25M.
if [[ ${BENCH_ONLY:-0} != 1 ]]; then
  for backend in "${backends[@]}"; do
    for dtype in fp32 fp16; do
      [[ $backend == pytorch* && $dtype == fp16 ]] && continue
      ncu --profile-from-start off --set full --section SourceCounters --import-source yes -f \
        -o "$out/ncu/${backend#cuda_}_${dtype}_25000000" \
        "$python" -m lab.profile interleave_arrays "$backend" "$dtype" n=25000000
    done
  done
fi

# The chart needs matplotlib; run this step wherever it is installed.
if "$python" -c "import matplotlib" 2>/dev/null; then
  "$python" -m problems.interleave_arrays.benchmarks.plot "$out"/device_*.json
fi
