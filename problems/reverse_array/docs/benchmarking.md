# Benchmarking

[← Reverse Array](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · **Benchmarking** · [Profiling](profiling.md)

Reverse Array uses the shared runner in [lab/bench.py](../../../lab/bench.py). Lab-wide
rules are in the [benchmark policy](../../../docs/benchmarking.md).

## Single run

```bash
python -m lab.bench reverse_array --copy-reference --peak-gb-s 960 \
  --output artifacts/reverse_array/device_1.json
python -m lab.report artifacts/reverse_array/device_1.json
```

- Each backend is checked against the PyTorch reference before it is timed.
- `--copy-reference` also times a device copy of the input (`copy_reference`) and adds
  `fraction_of_copy` to every row. Reversal moves the same bytes as a copy, so this is
  its practical speed limit.
- Narrow a run with `--backends`, `--dtypes`, `--warmup` and `--samples`.

## Benchmark sizes

`BENCH_CASES` in [cases.py](../cases.py): N = 1025 and 1M (launch- and cache-bound),
25,000,000 and 25,000,001 (the source's performance size, aligned and misaligned), and
100,000,000 and 100,000,001 (the contract maximum). The 100M sizes matter because the
kernel works in place: 25M FP16 elements are 50 MB and stay in the 64 MiB L2 between
calls, while a copy writes a second buffer and spills to DRAM. Compare DRAM behavior on
the 100M rows.

## Reproducing the recorded results

[sweep.sh](../benchmarks/sweep.sh) runs everything behind [results](results.md): the test
suite as a gate, three benchmark runs, the 20 Nsight Compute reports and the chart.

```bash
PYTHON=.venv/bin/python TORCH_CUDA_ARCH_LIST=12.0 \
  problems/reverse_array/benchmarks/sweep.sh artifacts/reverse_array 960
python -m problems.reverse_array.benchmarks.plot artifacts/reverse_array/device_*.json
```

The benchmark runs take under a minute; the Nsight reports a few more. The chart step
needs matplotlib and is skipped when it is missing.
