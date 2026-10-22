#!/usr/bin/env bash
# Run from the repository root on the NVIDIA host.
set -euo pipefail
out=${1:-artifacts/rainbow_table}
python=${PYTHON:-python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PWD/artifacts/torch_extensions}
mkdir -p "$out/ncu" "$out/sass"
backends=(pytorch pytorch_int32 pytorch_compile cuda_int64 cuda_u32 cuda_ilp2 cuda_ilp cuda_vec cuda_unroll cuda triton triton_unroll)
if [[ ${SKIP_BENCH:-0} != 1 ]]; then
  "$python" -m pytest -q problems/rainbow_table > "$out/tests.log" 2>&1
  for run in 1 2 3; do
    "$python" -m lab.bench rainbow_table --backends "${backends[@]}" --copy-reference --peak-gb-s 960 --output "$out/device_$run.json" > "$out/bench_$run.log" 2>&1
  done
fi
for backend in "${backends[@]}"; do
  for rounds in 1 10 100; do
    n=10000000
    [[ $rounds == 10 ]] && n=5000000
    extra=()
    # Eager has hundreds/thousands of launches. Capture the first representative
    # launch only; benchmark JSON records the complete backend's cost.
    [[ $backend == pytorch || $backend == pytorch_int32 ]] && extra=(--launch-count 1)
    ncu --profile-from-start off --set full --section SourceCounters --import-source yes \
      --metrics sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_aluheavy.sum,sm__inst_executed_pipe_fmaheavy.sum,sm__inst_executed_pipe_fmalite.sum --cache-control all "${extra[@]}" -f -o "$out/ncu/${backend}_r${rounds}" \
      "$python" -m lab.profile rainbow_table "$backend" int32 n="$n" rounds="$rounds" dist=source \
      > "$out/ncu/${backend}_r${rounds}.log" 2>&1
    report="$out/ncu/${backend}_r${rounds}.ncu-rep"
    [[ -f "${report}z" ]] && report="${report}z"
    ncu --import "$report" --page raw --csv \
      > "$out/ncu/${backend}_r${rounds}.csv"
    ncu --import "$report" --page source --print-source sass > "$out/ncu/${backend}_r${rounds}.sass"
  done
done
so="$TORCH_EXTENSIONS_DIR/gpu_lab_rainbow_table/gpu_lab_rainbow_table.so"
cuobjdump --dump-sass "$so" > "$out/sass/cuda.sass"
cuobjdump --dump-resource-usage "$so" > "$out/sass/resources.txt"
"$python" -m problems.rainbow_table.benchmarks.inspect_sass "$out/sass/cuda.sass" > "$out/sass/counts.json"
"$python" -m problems.rainbow_table.benchmarks.summarize "$out"
if "$python" -c "import matplotlib" 2>/dev/null; then
  "$python" -m problems.rainbow_table.benchmarks.plot "$out"
else
  echo "Captures complete; matplotlib is absent. Run benchmarks.plot locally on the downloaded artifacts."
fi
