#!/usr/bin/env bash
# Reproduce the recorded Matrix Copy results. Run from the repository root on an NVIDIA host.
# Usage: sweep.sh [output_dir] [peak_gb_s]
set -euo pipefail
out=${1:-artifacts/matrix_copy}
peak=${2:-960}
python=${PYTHON:-python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PWD/artifacts/torch_extensions}
mkdir -p "$out/ncu" "$out/sass"
# `cuda` is an alias of cuda_flat and is not timed separately.
kernels=(cuda_2d cuda_2d_colmajor cuda_flat cuda_vec cuda_vec_ilp2 cuda_vec_ilp4
  cuda_vec_gridstride cuda_stream triton triton_2d)
backends=(pytorch "${kernels[@]}")
metrics=gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum
metrics+=,lts__t_sector_hit_rate.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed
metrics+=,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
metrics+=,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
metrics+=,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
metrics+=,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread
metrics+=,launch__grid_size,launch__block_size,smsp__inst_executed.sum
metrics+=,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio

if [[ ${SKIP_BENCH:-0} != 1 ]]; then
  # Correctness gate: never time or profile a backend that fails its tests.
  "$python" -m pytest -q problems/matrix_copy > "$out/tests.log" 2>&1
  # Three complete runs, with a device-to-device copy of the input as the reference.
  for run in 1 2 3; do
    "$python" -m lab.bench matrix_copy --backends "${backends[@]}" --copy-reference \
      --peak-gb-s "$peak" --output "$out/device_$run.json" > "$out/bench_$run.log" 2>&1
  done
fi

# One launch per report. The PyTorch reference is a cudaMemcpyAsync, not a kernel, so ncu
# has nothing to capture for it. Cold: 4096² (the source's size), caches flushed, the
# DRAM-bound regime. Warm: 2048², caches kept, so the warmups leave the data in L2 as in
# the benchmark loop.
profile() { # name size cache-control backend dtype
  ncu --profile-from-start off --metrics "$metrics" --cache-control "$3" -f -o "$out/ncu/$1" \
    "$python" -m lab.profile matrix_copy "$4" "$5" m="$2" n="$2" > "$out/ncu/$1.log" 2>&1
  local report="$out/ncu/$1.ncu-rep"
  [[ -f "${report}z" ]] && report="${report}z"
  ncu --import "$report" --page raw --csv > "$out/ncu/$1.csv"
}
for backend in "${kernels[@]}"; do
  for dtype in fp32 fp16; do
    profile "${backend}_${dtype}_4096_cold" 4096 all "$backend" "$dtype"
    profile "${backend}_${dtype}_2048_warm" 2048 none "$backend" "$dtype"
  done
done

# Static SASS: CUDA from the built extension, Triton from its compiled kernels.
so="$TORCH_EXTENSIONS_DIR/gpu_lab_matrix_copy/gpu_lab_matrix_copy.so"
cuobjdump --dump-sass "$so" > "$out/sass/cuda.sass"
cuobjdump --dump-resource-usage "$so" > "$out/sass/resources.txt"
"$python" -m problems.matrix_copy.benchmarks.inspect_sass "$out/sass"

"$python" -m problems.matrix_copy.benchmarks.summarize "$out"
# The chart needs matplotlib; run this step wherever it is installed.
if "$python" -c "import matplotlib" 2>/dev/null; then
  "$python" -m problems.matrix_copy.benchmarks.plot "$out"
else
  echo "matplotlib is absent: run benchmarks.plot locally on the copied artifacts."
fi
