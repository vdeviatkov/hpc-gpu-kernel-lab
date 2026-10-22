# Benchmarking

[← Rainbow Table](../README.md) · [Results](results.md) · [Profiling](profiling.md)

Run on cuda.0 from `/home/vdeviatkov/Documents/hpc-gpu-kernel-lab`:

```bash
PYTHON=.venv/bin/python bash problems/rainbow_table/benchmarks/sweep.sh
```

The script gates on the full correctness suite, then performs **three complete
runs** of every backend plus a device-copy control. Each case has 25 warmups and
100 CUDA-event samples, with correctness checked before timing. Backend order is
seeded and shuffled per case. Compilation is excluded; output allocation and the
Python-dispatched call occur between events. Thus a short kernel's event interval
can include a CPU submission gap: this is not pure hardware kernel time.

Cases: N = 10,000/R = 10 (call/launch overhead); N = 5M/R = 10 (source); N = 10M
with R = 1,2,5,10,20,50,100 (headline sweep). Inputs are reused without flushing.
Reference allocation is outside timing; intermediates inside an eager backend
are included. The output is a fresh allocation per call. Copy uses matching
logical 8N traffic. No CPU timing fallback exists.

JSON contains environment, package versions, clock-related flags, repository state,
raw samples, correctness and run settings. Eager references have many intermediate
passes, so their logical effective bandwidth is not actual hardware bandwidth.
Compare run medians; the figure shows median of three medians, shaded min–max.

```bash
python -m problems.rainbow_table.benchmarks.plot artifacts/rainbow_table
```

The script then writes 36 Nsight reports (`.ncu-repz` on Nsight 2026.3) (three R cases for twelve backend names),
raw CSVs, per-instruction SASS exports and CUDA resource/disassembly files.
Eager reports are representative first-launch captures, explicitly not full traces.
All generated large artifacts stay under ignored `artifacts/`; the SVG and measured
summary remain tracked. Keep reports with the metadata that produced them.

If matplotlib is absent on the GPU host, render the chart locally after copying
the artifacts. `SKIP_BENCH=1` reruns profiling using existing benchmark JSON.
Create the compact tracked evidence with:

```bash
python -m problems.rainbow_table.benchmarks.summarize artifacts/rainbow_table
```
