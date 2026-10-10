#!/usr/bin/env bash
# Reproduce the recorded Reverse Array results. Run from the repository root on an NVIDIA
# host. Usage: sweep.sh [output_dir] [peak_gb_s]
set -euo pipefail
out=${1:-artifacts/reverse_array}
peak=${2:-960}
python=${PYTHON:-python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PWD/artifacts/torch_extensions}
mkdir -p "$out/ncu"

# Correctness gate: never time or profile a backend that fails its tests.
"$python" -m pytest -q problems/reverse_array

# Three complete benchmark runs of every backend, case and dtype, with the copy reference.
for i in 1 2 3; do
  "$python" -m lab.bench reverse_array --copy-reference --peak-gb-s "$peak" \
    --output "$out/device_$i.json"
done

# Full Nsight Compute reports (all sections plus per-instruction source counters) for one
# launch of each profiled backend at the DRAM-bound size, aligned and misaligned.
while read -r backend dtype n; do
  ncu --profile-from-start off --set full --section SourceCounters --import-source yes -f \
    -o "$out/ncu/${backend#cuda_}_${dtype}_${n}" \
    "$python" -m lab.profile reverse_array "$backend" "$dtype" "n=$n"
done <<LIST
cuda_pair fp32 100000000
cuda_pair fp32 100000001
cuda_pair fp16 100000000
cuda_pair fp16 100000001
cuda_vec fp16 100000000
cuda_tile fp32 100000000
cuda_tile fp32 100000001
cuda_tile fp16 100000000
cuda_tile fp16 100000001
cuda_tile_async fp32 100000000
cuda_tile_async fp16 100000001
triton_pair fp32 100000000
triton_pair fp32 100000001
triton_pair fp16 100000000
triton_pair fp16 100000001
triton_flip fp32 100000000
triton_flip fp32 100000001
triton_flip fp16 100000000
triton_flip fp16 100000001
pytorch fp32 100000000
LIST

# The chart needs matplotlib; run this step wherever it is installed.
if "$python" -c "import matplotlib" 2>/dev/null; then
  "$python" -m problems.reverse_array.benchmarks.plot "$out"/device_*.json
fi
