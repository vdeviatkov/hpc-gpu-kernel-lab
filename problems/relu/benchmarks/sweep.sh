#!/usr/bin/env bash
# Reproduce the recorded ReLU results. Run from the repository root on an NVIDIA host.
# Usage: sweep.sh [output_dir] [peak_gb_s]
set -euo pipefail
out=${1:-artifacts/relu}
peak=${2:-960}
python=${PYTHON:-python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PWD/artifacts/torch_extensions}
mkdir -p "$out"

# Three complete runs of every backend, case and dtype, with the copy reference.
for i in 1 2 3; do
  "$python" -m lab.bench relu --copy-reference --peak-gb-s "$peak" --output "$out/device_$i.json"
done

# Nsight Compute counters for one launch per backend, dtype and sign pattern.
metrics=gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed
metrics+=,smsp__inst_executed.sum,smsp__sass_inst_executed_op_global_st.sum
metrics+=,smsp__sass_branch_targets_threads_divergent.sum
metrics+=,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread
for dtype in fp32 fp16; do
  for backend in pytorch cuda_select cuda_branch cuda_fmax cuda_vec triton; do
    for dist in mixed alternating warp_blocks positive negative; do
      ncu --profile-from-start off --csv --metrics "$metrics" \
        "$python" -m problems.relu.benchmarks.profile "$backend" "$dtype" 25000000 "$dist" \
        > "$out/ncu_${backend}_${dtype}_${dist}.csv"
    done
  done
done

# The chart needs matplotlib; run this step wherever it is installed.
if "$python" -c "import matplotlib" 2>/dev/null; then
  "$python" -m problems.relu.benchmarks.plot "$out"/device_*.json
fi
