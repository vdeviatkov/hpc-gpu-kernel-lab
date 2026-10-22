# Results

[← Rainbow Table](../README.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md) · [SASS](sass.md) · [Roofline](roofline.md)

Measured 2026-10-22 UTC on cuda.0: RTX 5080, 84 SMs, compute capability 12.0,
driver 615.71.09, CUDA toolkit 13.4.92, PyTorch 2.14.1+cu130, Triton 3.8.0,
Nsight Compute 2026.3.1. A display server also uses this GPU. Three runs, 25
warmups and 100 samples per case/run; values below are medians of run medians.
Full raw data and environment are in `artifacts/rainbow_table/device_{1,2,3}.json`;
compact evidence is in [measured_summary.json](../benchmarks/measured_summary.json).

![Rounds sweep and integer roofline view](../figures/r_sweep.svg)

## Benchmark latency

Microseconds, complete dispatched call bracketed by CUDA events, including output
allocation and any submission gaps. Nsight kernel times below are a different
measurement. CUDA is an alias of cuda_u32; separate runs are a noise control.

| Backend | N=10M, R=1 | Source N=5M, R=10 | N=10M, R=100 | R=100 T rounds/s |
|---|---:|---:|---:|---:|
| pytorch | 4,646.192 | 15,092.432 | 437,502.823 | 0.0023 |
| pytorch_int32 | 1,241.424 | 3,368.096 | 124,731.827 | 0.0080 |
| pytorch_compile | 108.800 | 43.712 | 543.776 | 1.8390 |
| cuda_int64 | 100.896 | 43.488 | 608.736 | 1.6427 |
| cuda_u32 | 100.896 | 39.840 | 555.552 | 1.8000 |
| cuda_ilp2 | 100.896 | 39.712 | 526.912 | 1.8979 |
| cuda_ilp | 101.920 | 37.728 | 509.568 | 1.9624 |
| cuda_vec | 101.680 | 35.776 | 517.696 | 1.9316 |
| cuda_unroll | 100.880 | 39.744 | 543.232 | 1.8408 |
| cuda | 101.632 | 39.888 | 555.584 | 1.7999 |
| triton | 103.776 | 40.800 | 525.520 | 1.9029 |
| triton_unroll | 103.904 | 40.752 | 536.048 | 1.8655 |
| copy_reference | 97.632 | 15.296 | 97.696 | — |

## Interpretation

- **The headline transition is visible.** At N=10M the fused kernels initially
  deliver about 0.10 T rounds/s and approach 1.6–2.0 T rounds/s by R=20–100.
  With the best measured R=100 rate (1.962 T rounds/s), the nominal 960 GB/s
  logical roof intersects near R=16.4. Using the measured copy rate (~819 GB/s)
  instead gives R≈19.2. These are approximate model ridges, not fitted hardware
  peak measurements.
- **Four-chain ILP gives a modest compute-regime improvement.** cuda_ilp beats u32
  by 9.0% at R=100 (1.09× throughput). At the source size, vector ILP leads at
  35.776 µs, versus 39.840 µs for u32. Its advantage combines fewer memory
  instructions with independent chains; it cannot beat the DRAM ceiling at R=1.
- **int64 is slower, but not because of wide multiplies.** It is 9.6% slower than
  u32 at R=100; the compiler removed the high product and masks. The remaining
  wide shifts and setup raise the loop from 15 to 16 instructions per element.
- **Unrolling is a small gain here.** cuda_unroll improves on u32 by 2.2% at R=100,
  but emits an extra LOP3 per round. All specialized positive-R CUDA kernels use
  the same 10 static registers, so there is no observed unroll register growth.
- **Triton is competitive.** The runtime loop reaches 1.903 T rounds/s at R=100;
  static unroll reaches 1.866. Static unrolling is not automatically faster.
  Both passed the entire GPU correctness suite.
- **The eager curves flatten for a different reason.** At R=100 pytorch_int32 is
  ~237× slower than looped Triton; int64 eager is ~833× slower. Each round adds
  launches and intermediate-memory traffic. Their flat rounds/s is not the
  single-kernel integer ceiling.

## Cache and timing limitations

The N=10M copy control moves logical 80 MB in ~97.7 µs (819 GB/s); at N=5M it
moves 40 MB in 15.3 µs (~2.62 TB/s), evidence of L2 reuse at the source case.
Do not call this source-size result DRAM bandwidth. Repeated output allocation,
cache residency, base-clock profiling vs normal benchmark clocks, display use,
and event submission gaps can all change small differences. Shading on the chart
shows the min–max of three run medians; raw within-run samples remain available.

## Nsight Compute: cold-cache kernels

Durations in µs, utilization and occupancy in percent. Each pair uses N=10M.
These are one-launch captures with replay and clock control, so they are not
substitutes for the benchmark's call times.

| Backend | R=1 µs | R=1 DRAM % | R=100 µs | R=100 SM % | R=100 DRAM % | R=100 issue % | R=100 occupancy % | Runtime regs/thread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cuda_u32 | 59.33 | 89.1 | 578.46 | 91.3 | 10.1 | 91.7 | 96.1 | 16 |
| cuda_int64 | 58.62 | 89.8 | 633.86 | 88.9 | 9.3 | 89.1 | 96.9 | 16 |
| cuda_ilp2 | 56.77 | 80.8 | 548.03 | 94.2 | 10.7 | 87.2 | 97.4 | 18 |
| cuda_ilp | 49.41 | 94.2 | 530.02 | 96.1 | 11.0 | 85.4 | 97.2 | 30 |
| cuda_vec | 49.50 | 94.1 | 539.10 | 96.7 | 10.8 | 83.5 | 96.7 | 20 |
| cuda_unroll | 58.14 | 90.1 | 564.93 | 98.9 | 10.4 | 75.1 | 97.1 | 16 |
| triton | 48.96 | 94.5 | 548.93 | 95.5 | 10.8 | 86.7 | 97.2 | 16 |
| triton_unroll | 48.99 | 94.0 | 563.65 | 99.1 | 10.4 | 75.1 | 97.9 | 16 |
| pytorch_compile | 47.65 | 94.7 | 561.50 | 98.9 | 10.5 | 75.1 | 97.9 | 16 |

At R=1, DRAM is 81–95% utilized and SM throughput is only 12–23%. At R=100,
DRAM is ~9–11% and SM throughput is 89–99%. This supports the memory-to-compute
transition for the fused implementations. At source R=10, u32 reports 84.3% SM /
58.5% DRAM; vec reports a different memory-instruction count and independent chains.
The source profile is cold, while its benchmark enjoys warm-cache reuse.

### ILP vs more warps

| R=100 backend | Wait / issue | Math throttle / issue | Long scoreboard / issue |
|---|---:|---:|---:|
| cuda_u32 | 2.082 | 3.532 | 0.570 |
| cuda_ilp2 | 0.833 | 5.221 | 0.158 |
| cuda_ilp | 0.348 | 5.871 | 0.040 |
| cuda_vec | 0.365 | 6.035 | 0.035 |
| triton | 0.787 | 5.367 | 0.308 |
| triton_unroll | 0.844 | 6.656 | 0.388 |
| cuda_unroll | 2.672 | 5.454 | 0.746 |

ILP4 lowers wait from 2.08 to 0.35 while raising math throttle from 3.53 to 5.87.
Occupancy is already 96–97% for both scalar and ILP at R=100. In this comparison,
ILP exposes independent math without sacrificing meaningful residency. The
remaining pipe pressure explains the modest improvement rather than a 4× gain.
This is evidence for this configuration, not a universal claim that ILP beats
more warps; no different GPU or exhaustive block-size search was measured.

### Which pipe carries IMAD?

For scalar u32 at N=10M/R=100, FMA executes 157,187,508 warp instructions,
about **5.03 per element-round after multiplying by 32 / NR**. Four are the
prime IMADs and one is the HFMA2 constant materialization in each round; setup
accounts for the small excess. ALU executes 220,000,004 (~7.04 per round),
matching four LOP3 plus three SHF and setup. Thus the multiply work is carried
by the FMA path in this capture, despite being integer arithmetic. Four-chain
ILP reduces FMA work to ~4.51 per round by amortizing constants. Blackwell's
FMA-heavy and ALU-heavy counters describe related scheduling resources and
must not be blindly added to their parent counters.

### Register evidence

cuobjdump reports 10 static registers for scalar and unrolled CUDA, while Nsight
reports 16 runtime registers/thread for those launches. Keep those measurements
distinct; occupancy decisions use runtime allocation and hardware granularity.
ILP2/ILP4/vec use 18/30/20 runtime registers. Compiled PyTorch changes from
26 at R=1 to 16 at R=100; both explicit Triton variants use 16 here. Unrolling
increases instruction footprint substantially even when registers stay flat.

### Trace inventory and limitations

36 compressed `.ncu-repz` reports, each with a raw CSV and source/SASS export,
are available in `artifacts/rainbow_table/ncu/`. CUDA disassembly, static resource
usage and parsed loop counts are under `artifacts/rainbow_table/sass/`.
Eager references have first-launch-only reports; all fused variants have their
complete single launch. Missing imported PyTorch/CUDA library source does not
remove the SASS view. Per-instruction samples can be noisy in short kernels.
