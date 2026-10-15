#!/usr/bin/env bash
# Reproduce the recorded RGB to Grayscale results. Run from the repository root on an NVIDIA
# host. Usage: sweep.sh [output_dir] [peak_gb_s]
set -euo pipefail
out=${1:-artifacts/rgb_to_grayscale}
peak=${2:-960}
python=${PYTHON:-python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PWD/artifacts/torch_extensions}
mkdir -p "$out/ncu"
cuda=(cuda_pixel cuda_vec cuda_smem cuda_planar)
backends=(pytorch pytorch_matmul pytorch_compile "${cuda[@]}")

# Correctness gate: never time or profile a backend that fails its tests.
"$python" -m pytest -q problems/rgb_to_grayscale

# Three complete benchmark runs, with a copy of the image as the reference.
for i in 1 2 3; do
  "$python" -m lab.bench rgb_to_grayscale --backends "${backends[@]}" --copy-reference \
    --peak-gb-s "$peak" --output "$out/device_$i.json"
done

# Full Nsight Compute reports (all sections plus per-instruction source counters), one launch
# per report. lab.profile warms up first, so torch.compile's compilation is never captured.
profile() { # name size cache-control backend dtype
  ncu --profile-from-start off --set full --section SourceCounters --import-source yes -f \
    --cache-control "$3" -o "$out/ncu/$1" \
    "$python" -m lab.profile rgb_to_grayscale "$4" "$5" h="$2" w="$2" dist=int
}
for backend in "${backends[@]}"; do
  for dtype in fp32 fp16; do
    [[ $backend == pytorch && $dtype == fp16 ]] && continue
    # 4096 x 4096 (the lab extension), caches flushed: the DRAM-bound regime.
    profile "${backend#cuda_}_${dtype}_4096" 4096 all "$backend" "$dtype"
    # 2048 x 2048 (the source's size), caches kept: the image stays in L2 after the warmup,
    # as in the benchmark loop.
    [[ $backend == pytorch || $backend == pytorch_matmul ]] && continue
    profile "${backend#cuda_}_${dtype}_2048_warm" 2048 none "$backend" "$dtype"
  done
done

# The chart needs matplotlib; run this step wherever it is installed.
if "$python" -c "import matplotlib" 2>/dev/null; then
  "$python" -m problems.rgb_to_grayscale.benchmarks.plot "$out"/device_*.json
fi
