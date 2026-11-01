#!/usr/bin/env bash
# Reproduce the recorded Matrix Transpose results. Run from the repository root on an NVIDIA host.
# Usage: sweep.sh [output_dir] [peak_gb_s]
set -euo pipefail
out=${1:-artifacts/matrix_transpose}
peak=${2:-960}
python=${PYTHON:-python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PWD/artifacts/torch_extensions}
mkdir -p "$out/ncu/full" "$out/nsys" "$out/sass"
# `cuda` is an alias of cuda_smem_pad and is not timed separately; Triton is a stub.
kernels=(cuda_naive_row cuda_naive_col cuda_smem cuda_smem_pad cuda_smem_swizzle cuda_vec_ilp)
metrics=gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum
metrics+=,lts__t_sector_hit_rate.pct,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
metrics+=,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
metrics+=,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
metrics+=,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
metrics+=,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
metrics+=,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
metrics+=,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
metrics+=,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum
metrics+=,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread
metrics+=,launch__shared_mem_per_block_static,launch__grid_size,launch__block_size
metrics+=,smsp__inst_executed.sum
metrics+=,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio

if [[ ${SKIP_BENCH:-0} != 1 ]]; then
  # Correctness gate: never time or profile a backend that fails its tests.
  "$python" -m pytest -q problems/matrix_transpose > "$out/tests.log" 2>&1
  for tool in memcheck racecheck synccheck initcheck; do
    compute-sanitizer --tool "$tool" --target-processes all --error-exitcode 1 \
      "$python" -m pytest -q -p no:cacheprovider problems/matrix_transpose -k "cuda" \
      > "$out/sanitizer_$tool.log" 2>&1
  done
  # Three complete runs of every kernel with memcpy as the reference, then 07's cuda_vec
  # and memcpy on the same cases as the copy control. Both are plain copies (not
  # transposes): the bandwidth ceiling, not competitors.
  for run in 1 2 3; do
    "$python" -m lab.bench matrix_transpose --backends pytorch "${kernels[@]}" --dtypes fp32 fp16 \
      --copy-reference --peak-gb-s "$peak" --output "$out/device_$run.json" > "$out/bench_$run.log" 2>&1
    "$python" -m problems.matrix_transpose.benchmarks.copy_control --backends pytorch cuda_vec \
      --dtypes fp32 fp16 --copy-reference --peak-gb-s "$peak" --output "$out/copy_$run.json" \
      > "$out/copy_$run.log" 2>&1
  done
fi

# One launch per report. Cold: the source's 7000 x 6000, caches flushed, DRAM-bound. Warm:
# 2048², caches kept, so the data is in L2 as in the benchmark loop.
profile() { # name problem backend dtype m n cache-control [extra ncu args...]
  local name=$1 problem=$2 backend=$3 dtype=$4 m=$5 n=$6 cache=$7
  shift 7
  ncu --profile-from-start off --cache-control "$cache" "$@" -f -o "$out/ncu/$name" \
    "$python" -m lab.profile "$problem" "$backend" "$dtype" m="$m" n="$n" > "$out/ncu/$name.log" 2>&1
  local report="$out/ncu/$name.ncu-rep"
  [[ -f "${report}z" ]] && report="${report}z"
  ncu --import "$report" --page raw --csv > "$out/ncu/$name.csv"
}
for dtype in fp32 fp16; do
  for backend in "${kernels[@]}"; do
    profile "${backend}_${dtype}_7000x6000_cold" matrix_transpose "$backend" "$dtype" 7000 6000 all \
      --metrics "$metrics"
    profile "${backend}_${dtype}_2048_warm" matrix_transpose "$backend" "$dtype" 2048 2048 none \
      --metrics "$metrics"
    # Every section, with source correlation, for manual inspection in the Nsight Compute GUI.
    profile "full/${backend}_${dtype}_7000x6000" matrix_transpose "$backend" "$dtype" 7000 6000 all \
      --set full --import-source yes
  done
  profile "copy_cuda_vec_${dtype}_7000x6000_cold" matrix_copy cuda_vec "$dtype" 7000 6000 all \
    --metrics "$metrics"
  profile "copy_cuda_vec_${dtype}_2048_warm" matrix_copy cuda_vec "$dtype" 2048 2048 none \
    --metrics "$metrics"
done

# A timeline of every kernel and both copy controls on 7000 x 6000.
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite true -o "$out/nsys/timeline" \
  "$python" -m problems.matrix_transpose.benchmarks.timeline > "$out/nsys/timeline.log" 2>&1
nsys stats --report cuda_gpu_kern_sum,nvtx_sum --format csv --force-export true \
  --output "$out/nsys/timeline" "$out/nsys/timeline.nsys-rep" >> "$out/nsys/timeline.log" 2>&1

# Static SASS and resource usage of the built extension.
so="$TORCH_EXTENSIONS_DIR/gpu_lab_matrix_transpose/gpu_lab_matrix_transpose.so"
cuobjdump --dump-sass "$so" > "$out/sass/cuda.sass"
cuobjdump --dump-resource-usage "$so" > "$out/sass/resources.txt"

"$python" -m problems.matrix_transpose.benchmarks.summarize "$out"
