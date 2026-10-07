# Benchmarking

[← ReLU](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · **Benchmarking** · [Profiling](profiling.md)

ReLU uses the shared runner in [lab/bench.py](../../../lab/bench.py). Lab-wide rules are
in the [benchmark policy](../../../docs/benchmarking.md).

## Single run

```bash
python -m lab.bench relu --copy-reference --peak-gb-s 960 --output artifacts/relu.json
python -m lab.report artifacts/relu.json
```

- Each backend is checked against `torch.relu` before it is timed.
- Cases come from `BENCH_CASES` in [cases.py](../cases.py): mixed signs at N = 1025,
  1M and 25M, plus positive, negative, alternating and warp-block signs at 25M.
- `--copy-reference` also times a device copy of the input (`copy_reference`) and adds
  `fraction_of_copy` to every row. ReLU moves the same bytes as a copy, so this is its
  practical speed limit.
- `--peak-gb-s 960` adds `fraction_of_peak` against the RTX 5080's DRAM peak.
- Narrow the run with `--backends`, `--dtypes`, `--warmup`, `--samples`.

Timing is device scope: CUDA events around one call, synchronized after each sample.
Output allocation and Python dispatch are inside the timed region, which matters only
for small inputs ([results](results.md#small-inputs)).

## Reproducing the recorded results

[sweep.sh](../benchmarks/sweep.sh) runs everything behind [results](results.md): three
benchmark runs, Nsight Compute for every backend, dtype and sign pattern, and the chart.

```bash
PYTHON=.venv/bin/python TORCH_CUDA_ARCH_LIST=12.0 \
  problems/relu/benchmarks/sweep.sh artifacts/relu 960
python -m problems.relu.benchmarks.plot artifacts/relu/device_*.json
```

The chart step needs matplotlib; `sweep.sh` skips it when matplotlib is missing.
