#!/usr/bin/env bash
# Reproduce the recorded 1D Convolution results. Run from the repository root on an NVIDIA host.
# Usage: sweep.sh [output_dir] [peak_gb_s]
set -euo pipefail
out=${1:-artifacts/convolution_1d}
peak=${2:-960}
python=${PYTHON:-python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PWD/artifacts/torch_extensions}
mkdir -p "$out/ncu" "$out/sass"
# `cuda` is an alias of cuda_smem_reg and is not timed separately. cuDNN (pytorch) is timed
# only in the N sweep: its FP16 path falls back to a generic kernel from K = 255 and would
# take minutes per K-sweep point (docs/design.md, §4); it still serves as every case's
# correctness reference.
kernels=(cuda_naive cuda_const cuda_smem cuda_smem_reg cuda_large_k cuda_unrolled triton)
n_profile=16777216
metrics=gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second
metrics+=,dram__bytes_op_read.sum,dram__bytes_op_write.sum
metrics+=,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
metrics+=,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct
metrics+=,l1tex__throughput.avg.pct_of_peak_sustained_active
metrics+=,l1tex__lsuin_requests.sum,l1tex__lsuin_requests.avg.pct_of_peak_sustained_active
metrics+=,l1tex__data_pipe_lsu_wavefronts.sum
metrics+=,l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_active
metrics+=,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
metrics+=,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
metrics+=,smsp__inst_executed_pipe_fma.sum,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active
metrics+=,smsp__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_active
metrics+=,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
metrics+=,launch__registers_per_thread,launch__occupancy_limit_registers
metrics+=,sm__maximum_warps_per_active_cycle_pct,sm__warps_active.avg.pct_of_peak_sustained_active
metrics+=,launch__grid_size,launch__block_size

if [[ ${SKIP_BENCH:-0} != 1 ]]; then
  # Correctness gate: never time or profile a backend that fails its tests.
  "$python" -m pytest -q -p no:cacheprovider problems/convolution_1d > "$out/tests.log" 2>&1
  for tool in memcheck racecheck synccheck initcheck; do
    compute-sanitizer --tool "$tool" --target-processes all --error-exitcode 1 \
      "$python" -m pytest -q -p no:cacheprovider problems/convolution_1d -k "cuda or triton" \
      > "$out/sanitizer_$tool.log" 2>&1
  done
  run=(-m problems.convolution_1d.benchmarks.run)
  for r in 1 2 3; do
    "$python" -m problems.convolution_1d.benchmarks.fma_peak "$out/fma_peak_$r.json" \
      > "$out/fma_peak_$r.log" 2>&1
    "$python" "${run[@]}" ksweep --backends "${kernels[@]}" --dtypes fp32 fp16 --copy-reference \
      --peak-gb-s "$peak" --output "$out/ksweep_$r.json" > "$out/ksweep_$r.log" 2>&1
    "$python" "${run[@]}" nsweep --backends pytorch "${kernels[@]}" --dtypes fp32 fp16 \
      --copy-reference --peak-gb-s "$peak" --output "$out/nsweep_$r.json" > "$out/nsweep_$r.log" 2>&1
    "$python" "${run[@]}" rsweep --dtypes fp32 fp16 --peak-gb-s "$peak" \
      --output "$out/rsweep_$r.json" > "$out/rsweep_$r.log" 2>&1
    # 07's cuda_vec on matrices of N elements: the bandwidth ceiling, not a convolution.
    "$python" "${run[@]}" copy --backends pytorch cuda_vec --dtypes fp32 fp16 --peak-gb-s "$peak" \
      --output "$out/copy_$r.json" > "$out/copy_$r.log" 2>&1
  done
fi

# One launch per report, N = 2^24, caches flushed (DRAM-cold) as in the K sweep's large-N
# regime.
profile() { # name backend dtype n k [capture args...]
  local name=$1 backend=$2 dtype=$3 n=$4 k=$5
  shift 5
  ncu --profile-from-start off --cache-control all --metrics "$metrics" -f -o "$out/ncu/$name" \
    "$python" -m problems.convolution_1d.benchmarks.capture "$backend" "$dtype" "$n" "$k" "$@" \
    > "$out/ncu/$name.log" 2>&1
  local report="$out/ncu/$name.ncu-rep"
  [[ -f "${report}z" ]] && report="${report}z"
  ncu --import "$report" --page raw --csv > "$out/ncu/$name.csv"
}
for dtype in fp32 fp16; do
  for k in 3 31 255 2047; do
    for backend in "${kernels[@]}"; do
      [[ $backend == cuda_unrolled && $k != 3 && $k != 31 ]] && continue
      profile "${backend}_${dtype}_k${k}" "$backend" "$dtype" "$n_profile" "$k"
    done
  done
  # Register pressure against R (outputs per thread) at the ridge-point filter.
  for r in 2 4 8 16; do
    profile "cuda_smem_reg_r${r}_${dtype}_k255" cuda_smem_reg "$dtype" "$n_profile" 255 \
      --reg-outputs "$r"
  done
  profile "copy_cuda_vec_${dtype}" copy_cuda_vec "$dtype" "$n_profile" 1
done

# Halo overhead: cuda_large_k global-load sectors against the model chunks + 2K / span
# (x halo plus filter), with span = 256·R outputs per block varied through R. FP32 only.
halo_metrics=gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
halo_metrics+=,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_srcunit_tex_op_read.sum
mkdir -p "$out/halo"
for r in 2 4 8 16; do
  for k in 31 255 1023 2047; do
    name="halo_r${r}_k${k}"
    ncu --profile-from-start off --cache-control all --metrics "$halo_metrics" -f \
      -o "$out/halo/$name" "$python" -m problems.convolution_1d.benchmarks.capture cuda_large_k \
      fp32 "$n_profile" "$k" --reg-outputs "$r" > "$out/halo/$name.log" 2>&1
    report="$out/halo/$name.ncu-rep"
    [[ -f "${report}z" ]] && report="${report}z"
    ncu --import "$report" --page raw --csv > "$out/halo/$name.csv"
  done
done

# Static SASS and resource usage of the built extension.
so="$TORCH_EXTENSIONS_DIR/gpu_lab_convolution_1d/gpu_lab_convolution_1d.so"
cuobjdump --dump-sass "$so" > "$out/sass/cuda.sass"
cuobjdump --dump-resource-usage "$so" > "$out/sass/resources.txt"

"$python" -m problems.convolution_1d.benchmarks.summarize "$out"
