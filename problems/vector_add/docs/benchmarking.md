# Benchmarking

[← Vector Addition](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · **Benchmarking** · [Profiling](profiling.md)

Lab-wide rules are in the [benchmark policy](../../../docs/benchmarking.md).

## Single run

```bash
python -m problems.vector_add.benchmarks.run --peak-gb-s 960 --output artifacts/vector_add.json
python -m problems.vector_add.benchmarks.report artifacts/vector_add.json
```

Each backend passes correctness before warmup/timing. Default sizes are 1, 1025,
1,048,576 and 25,000,000; dtypes are FP32/FP16/BF16. BF16 benchmarks require SM80+
by policy. Requested missing packages and build/runtime errors abort the run;
select available implementations with `--backends`.

| Option | Effect |
|---|---|
| `--scope device\|api` | Timing scope (see below); default `device` |
| `--sizes`, `--dtypes`, `--backends` | Case matrix |
| `--threads 128\|256\|512` | CUDA block size |
| `--block-size`, `--num-warps` | Triton tile and warp count |
| `--offset N` | Element offset for all buffers (misalignment test) |
| `--peak-gb-s X` | Sourced hardware ceiling; adds `fraction_of_peak`. Nothing is guessed without it |
| `--no-reference` | Skip the `copy_reference` row |
| `--warmup`, `--samples`, `--seed` | Defaults 25, 100, 2026 |
| `--notes` | Free text for clocks, load and thermal state |
| `--profile` | Profiler capture mode; see [profiling](profiling.md) |

## Timing scopes

- **device:** preallocated output, CUDA events on the current stream, synchronized
  after each sample. This measures stream elapsed time around an eager call;
  host submission gaps can enter it. It is not isolated kernel duration.
- **api:** host-observed allocation, dispatch and completion. PyTorch/CUDA/Triton
  synchronize the selected device; JAX waits on its result. Transfers and JIT/build
  are excluded. Tiny comparisons include runtime-specific completion costs.
  JAX receives the exact GPU inputs through DLPack, never through a CPU fallback.

Never mix these scopes in a speedup claim. Backend order is shuffled per case with
a fixed seed; repeat whole runs to assess noise.

## Reproducing the recorded results

[sweep.sh](../benchmarks/sweep.sh) runs the full matrix behind [results](results.md):
three default runs, FP32/FP16 size sweeps over 13 sizes, the API scope with JAX, and
a misaligned run. [plot.py](../benchmarks/plot.py) regenerates the overview chart.

```bash
PYTHON=.venv/bin/python TORCH_CUDA_ARCH_LIST=12.0 \
  problems/vector_add/benchmarks/sweep.sh artifacts/vector_add 960
python -m problems.vector_add.benchmarks.plot \
  artifacts/vector_add/sizes_fp32.json artifacts/vector_add/sizes_fp16.json
```

## Output format

One JSON file per run (`schema_version: 1`):

- `environment`: GPU, compute capability, SM count, L2 size, driver/toolkit/runtime,
  installed packages, source commit and dirty state, relevant environment variables.
- `settings`: all options, cache and allocation policy.
- `results`: one row per backend/size/dtype with raw microsecond samples,
  p20/p50/p80, mean/stddev, effective GB/s, elements/s, `fraction_of_copy`,
  `fraction_of_peak`, `speedup_vs_pytorch`, and the vector path actually taken
  (`cuda_vector_path`).
