# Benchmarking

[← 1D Convolution](../README.md) · [Design](design.md) · [Results](results.md) · [Testing](testing.md) ·
**Benchmarking** · [Profiling](profiling.md)

1D Convolution uses the shared runner in [lab/bench.py](../../../lab/bench.py) through a thin
driver, [run.py](../benchmarks/run.py), which swaps `cases.BENCH_CASES` for one of the
study's sweeps. Lab-wide rules are in the [benchmark policy](../../../docs/benchmarking.md).

## Reproduce

On cuda.0, from a checkout of this repository:

```bash
PYTHON=.venv/bin/python bash problems/convolution_1d/benchmarks/sweep.sh artifacts/convolution_1d 960
# where matplotlib is installed, after copying artifacts/convolution_1d:
python -m problems.convolution_1d.benchmarks.plot artifacts/convolution_1d
```

[sweep.sh](../benchmarks/sweep.sh) runs, in order:

1. **Correctness gate.** The full test suite (`tests.log`), then Compute Sanitizer memcheck,
   racecheck, synccheck and initcheck over the CUDA and Triton tests (`sanitizer_*.log`; see
   [testing](testing.md)). A failure stops the sweep.
2. **Three complete runs**, each of:
   - [fma_peak.py](../benchmarks/fma_peak.py): the FP32 FFMA ceiling and the SM clock it
     sustains (`fma_peak_{1,2,3}.json`), the roofline's compute roof.
   - **K sweep** (`ksweep_*.json`): N = 2²⁴, K = 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023,
     2047, every CUDA kernel and Triton, FP32 and FP16. `cuda_unrolled` runs only at K = 3,
     7 and 31.
   - **N sweep** (`nsweep_*.json`): K = 31, N = 256 Ki to 64 Mi across the 64 MiB L2, the
     same kernels plus cuDNN (`pytorch`).
   - **R sweep** (`rsweep_*.json`): `cuda_smem_reg` with R = 2, 4, 8, 16 outputs per thread at
     K = 31, 255 and 2047, N = 2²⁴.
   - **Copy control** (`copy_*.json`): 07 · Matrix Copy's `cuda_vec` and `a.clone()` on
     (N / 1024) × 1024 matrices for every N above: 2·N·e bytes, the same logical traffic as
     a convolution's (2N + 1)·e. These are plain copies, the bandwidth ceiling, not
     competitors.
3. **Nsight Compute**: 62 single-launch reports with the metric set in
   [profiling](profiling.md), then 16 halo-overhead reports.
4. **Static SASS** and resource usage of the CUDA extension (`sass/`).
5. [summarize.py](../benchmarks/summarize.py), which writes the tracked
   [measured_summary.json](../benchmarks/measured_summary.json) and prints the tables in
   [results](results.md). `SKIP_BENCH=1` reuses existing benchmark JSON and only re-profiles.

## What one sample measures

- Each backend is checked against the PyTorch reference on the same inputs before it is
  timed; a mismatch stops the run.
- A sample is one call through the public API, timed with CUDA events on the current stream
  (`--scope device`), after 25 warm-up calls; 100 samples per case and backend. Each table
  value is the median of the three run medians.
- Inputs are reused between samples and the output is allocated inside the call, as a user
  would see it. The previous output is still referenced while the next is allocated, so the
  working set between samples is about x plus two outputs, 3·N·e bytes. The L2 boundary in
  the N sweep is therefore expected between the N where 3·N·e reaches 64 MiB and the N where
  one call's 2·N·e does: N ≈ 5.6 M to 8.4 M (FP32), 11.2 M to 16.8 M (FP16).
- **cuDNN is timed only in the N sweep (K = 31).** Its FP16 path falls back to a generic
  kernel from K = 255 (861 ms at N = 25 M, [design §4](design.md#4-triton-the-naive-stencil-measured)),
  so a K sweep would take hours; it still serves as every case's correctness reference.
- **TFLOP/s** counts 2·K·(N − K + 1) useful FLOPs. **Effective GB/s** divides the minimum
  logical traffic, (2N + 1)·e, by time. Neither is a hardware counter; Nsight Compute
  provides measured DRAM bytes and pipe utilization.
- Clocks are not locked. `fma_peak.py` records the clock the SM sustains under FFMA load in
  each run, and the roofline uses that run's measured peak.
