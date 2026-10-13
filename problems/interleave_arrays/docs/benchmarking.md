# Benchmarking

[← Interleave Arrays](../README.md) · [Design](design.md) · [Results](results.md) ·
[Testing](testing.md) · **Benchmarking** · [Profiling](profiling.md)

Interleave Arrays uses the shared runner in [lab/bench.py](../../../lab/bench.py). Lab-wide
rules are in the [benchmark policy](../../../docs/benchmarking.md).

## Single run

```bash
python -m lab.bench interleave_arrays \
  --backends pytorch pytorch_stack cuda_input cuda_output cuda_pair_store cuda_vec \
  --copy-reference --peak-gb-s 960 --output artifacts/interleave_arrays/device_1.json
python -m lab.report artifacts/interleave_arrays/device_1.json
```

- Each backend is checked against the PyTorch reference before it is timed.
- `--copy-reference` times a device copy of **every** input (here `a` and `b`, one copy
  each) and adds `fraction_of_copy` to every row. That moves exactly the bytes the kernels
  move: both inputs read once, `2N` elements written. At small sizes the two copies are two
  launches, so compare small sizes in µs rather than percent of copy.
- Leave out `--backends` to include `triton`, which reports "not implemented" while Triton
  is deferred.

## Benchmark sizes

`BENCH_CASES` in [cases.py](../cases.py): N = 1025 and 1M (launch- and cache-bound),
25,000,000 (the source's performance size) and 50,000,000 (the contract maximum). Out of
place, even 25M FP16 moves 200 MB, far more than the 64 MiB L2, so both large sizes are
DRAM-bound for every dtype.

## Reproducing the recorded results

[sweep.sh](../benchmarks/sweep.sh) runs everything behind [results](results.md): the test
suite as a gate, three benchmark runs, the 10 Nsight Compute reports and the chart.

```bash
PYTHON=.venv/bin/python TORCH_CUDA_ARCH_LIST=12.0 \
  problems/interleave_arrays/benchmarks/sweep.sh artifacts/interleave_arrays 960
python -m problems.interleave_arrays.benchmarks.plot artifacts/interleave_arrays/device_*.json
```

The benchmark runs take under a minute; the Nsight reports a few more. The chart step needs
matplotlib and is skipped when it is missing.
