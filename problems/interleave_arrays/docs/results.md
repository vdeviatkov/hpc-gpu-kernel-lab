# Results

[← Interleave Arrays](../README.md) · [Design](design.md) · **Results** ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

**Benchmark session:** 2026-10-23 PDT (2026-10-24 UTC), cuda.0, RTX 5080,
CUDA toolkit 13.4.92, PyTorch 2.14.1+cu130, Triton 3.8.0, driver 615.71.09.
**All backends, including Triton, were remeasured together.** Current benchmark
tables use the median of three run medians, each from 100 CUDA-event samples after
25 warmups. Timing includes output allocation and call submission; compilation
is excluded. Inputs are reused, clocks are unlocked, and the desktop uses the GPU.
Each benchmark case passed its correctness gate before timing.

The updated chart uses this same session. The older CUDA/PyTorch Nsight analysis
is retained in [Why the variants differ](#why-the-variants-differ), explicitly
separate from these new call timings. Compact per-run evidence and source hashes:
[measured_summary.json](../benchmarks/measured_summary.json). Raw samples and
full environment are in `artifacts/perf_20261023/interleave_arrays/device_{1,2,3}.json`.

## DRAM-bound sizes

Latency in µs (lower is better); parentheses give the median fraction of a
same-run copy of both inputs. Copy and interleave move the same logical bytes.

| N = 25,000,000 | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| copy reference | 488.896 | 244.256 | 244.320 |
| `pytorch` | 1384.544 (35.2%) | 691.168 (35.3%) | 690.208 (35.4%) |
| `pytorch_stack` | 1363.952 (35.8%) | 691.392 (35.3%) | 689.760 (35.4%) |
| `cuda_input` | 497.008 (98.3%) | 249.472 (97.9%) | 249.472 (97.9%) |
| `cuda_output` | 493.152 (99.1%) | 403.040 (60.6%) | 403.088 (60.6%) |
| `cuda_pair_store` | 492.976 (99.0%) | 249.440 (97.9%) | 248.352 (98.4%) |
| `cuda_vec` | 497.264 (98.3%) | 249.408 (97.9%) | 249.456 (97.9%) |
| `triton` | 498.240 (98.1%) | 251.248 (97.2%) | 250.608 (97.5%) |

| N = 50,000,000 | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| copy reference | 980.592 | 488.976 | 488.048 |
| `pytorch` | 2777.248 (35.4%) | 1376.736 (35.4%) | 1372.256 (35.6%) |
| `pytorch_stack` | 2771.488 (35.4%) | 1398.304 (35.0%) | 1394.800 (35.0%) |
| `cuda_input` | 983.088 (99.7%) | 495.952 (98.6%) | 495.232 (98.5%) |
| `cuda_output` | 989.056 (99.1%) | 796.176 (61.4%) | 796.144 (61.3%) |
| `cuda_pair_store` | 978.736 (100.0%) | 493.152 (98.8%) | 495.136 (98.7%) |
| `cuda_vec` | 989.392 (99.1%) | 497.184 (98.3%) | 496.240 (98.4%) |
| `triton` | 992.752 (98.8%) | 497.168 (98.2%) | 497.024 (98.4%) |

Triton reaches **97–98% of copy speed** at N=25M: 498.240 µs FP32,
251.248 µs FP16 and 250.608 µs BF16. It is within about 1% of `cuda_vec` there.
At N=50M it takes 992.752 / 497.168 / 497.024 µs, again about 98% of copy speed.
The two-byte `cuda_output` disadvantage remains; Triton's contiguous output tile
avoids that mapping's performance loss without requiring aligned inputs.

Triton's run-median range at N≥25M is at most **0.4%** of its median. These small
CUDA/Triton differences are not evidence of a large kernel-design advantage.

## Why the variants differ

**Historical profiler evidence:** the Nsight counters and kernel-only timings in
this section are from the earlier CUDA 13.2 CUDA/PyTorch study. Triton was not
captured in that study; the new benchmark tables above do not imply new Nsight
measurements.

Nsight Compute, one launch per report, N = 25M ([how to capture and read](profiling.md)).
ncu locks clocks and flushes caches, so its durations differ from the table above; compare
the kernels with each other. A memory-bound kernel can lose to a copy in four ways, checked
in order.

### 1. Wasted bytes: partial-sector writes

| FP32 | DRAM read | DRAM write | Store sectors/request (ideal) | L2 write hit rate |
|---|---:|---:|---|---:|
| *lower bound (`2·N·s` each way; no kernel can move less)* | *200 MB* | *200 MB* | — | — |
| `cuda_input` | 200 MB | 160 MB | 8 (4) | 33% |
| `cuda_pair_store` | 200 MB | 161 MB | 8 (8) | 0% |
| `cuda_vec` | 200 MB | 162 MB | 32 (32) | 30% |
| `pytorch` (2 kernels) | **2 × 287 MB** | 2 × 185 MB | 8 (4) | 0% |
| `pytorch_stack` | **589 MB** | 392 MB | 32 | 58% |

The italic row is computed, not measured: every input element read once and every output
element written once. Measured writes below it are output still in L2 when the kernel
ends, written to DRAM just after.

Memory writes whole 32-byte sectors back to DRAM; a sector written only partly is first
read from DRAM and merged. `cuda_input` touches twice the sectors it fills per store
instruction, yet reads exactly its 200 MB of input: the thread's second store completes
each sector in L2 (33% of writes hit a sector just started), so DRAM sees full sectors.
PyTorch's reference runs two kernels, `out[0::2] = a` and `out[1::2] = b`; each writes half
of every sector, so each must read the old sector from DRAM first (≈ 187 MB of extra reads
per pass) and write it back. That is about 2.4× the lower bound (944 MB measured
against 400 MB), and each pass keeps DRAM only 80% busy against ~88% for the CUDA kernels,
so the time grows a little more: 2.8× a copy. `torch.stack` is implemented as a
concatenation kernel whose blocks write the `a` and `b` columns separately, so it suffers
the same read-modify-write.

### 2. Is DRAM kept busy?

| N = 25M | DRAM busy | Active lanes per memory instruction | Load bytes per request |
|---|---:|---:|---:|
| `cuda_input` FP16 | 89.3% | 32 | 64 |
| `cuda_pair_store` FP16 | 89.1% | 32 | 64 |
| `cuda_vec` FP16 | 88.5% | 32 | 512 |
| `cuda_output` FP32 | 89.3% | 16 (loads), 32 (store) | 64 |
| **`cuda_output` FP16** | **48.0%** | **16 (loads and stores)** | **32** |

`cuda_output` moves no extra bytes (100 MB read in FP16, the lower bound); its requests are too
small. Each lane reads from `a` or `b` depending on whether its index is even, so the
compiler issues two predicated loads with 16 active lanes each; in FP16 it also splits the
store into two half-warp stores. A 16-lane FP16 request carries 32 bytes, one sector, too
little to keep enough bytes in flight, and DRAM idles half the time. In FP32 the same
half-warp loads carry 64 bytes and the store uses the full warp, which is enough.
`cuda_input` FP16 is fine (89%) because every instruction uses all 32 lanes and each thread
has two independent loads in flight, twice the bytes per warp of a single-load kernel like
ReLU's scalar FP16 version (67% busy).

### 3. Is another unit saturated?

Issue slots are 2–13% busy everywhere. Only `pytorch_stack` hits another limit: its stores
are spread 32 bytes apart per lane (32 sectors per request), and warps wait mostly on a
full memory-instruction queue (LG throttle, 272 cycles per issued instruction). `cuda_vec`
shows a small LG throttle (38–47) without effect: it keeps DRAM 87–89% busy.

### 4. Overhead

| FP32 | `cuda_input` | `cuda_output` | `cuda_pair_store` | `cuda_vec` |
|---|---:|---:|---:|---:|
| Instructions per element | 0.72 | 1.75 | 0.69 | 0.30 |
| Registers per thread | 16 | 16 | 16 | 26 |
| Warps active | 84% | 84% | 83% | 80% |

A 6× spread in instruction count produces no difference in FP32 time: when DRAM is the
limit, instructions are almost free.

## Small inputs

Complete-call latency in µs. The two-copy reference uses two launches, so a
fraction-of-copy comparison is less useful here than absolute latency.

| N = 1,025 | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| copy reference | 8.848 | 8.704 | 8.800 |
| `pytorch` | 10.832 | 10.784 | 10.816 |
| `pytorch_stack` | 7.648 | 7.648 | 7.680 |
| `cuda_input` | 10.656 | 10.432 | 10.400 |
| `cuda_output` | 10.528 | 10.368 | 10.368 |
| `cuda_pair_store` | 10.592 | 10.304 | 10.432 |
| `cuda_vec` | 10.656 | 10.432 | 10.432 |
| `triton` | 13.216 | 13.120 | 13.216 |

| N = 1,048,576 | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| copy reference | 8.896 | 9.024 | 8.960 |
| `pytorch` | 16.256 | 13.056 | 12.784 |
| `pytorch_stack` | 25.456 | 17.376 | 17.408 |
| `cuda_input` | 14.336 | 14.256 | 14.256 |
| `cuda_output` | 14.304 | 16.352 | 20.272 |
| `cuda_pair_store` | 14.256 | 12.256 | 12.256 |
| `cuda_vec` | 14.368 | 12.224 | 12.256 |
| `triton` | 15.296 | 13.312 | 13.248 |

Triton takes about 13 µs at N=1025; fixed call and launch costs dominate.

## Decisions

- **Triton is a competitive portable option:** about 98% of copy speed at large
  sizes, close to the CUDA kernels across all three dtypes.
- **`cuda_pair_store` remains the recommended CUDA kernel:** at copy speed for every dtype, full
  sectors per store instruction, no alignment requirement (the output is freshly
  allocated), and the simplest code. `cuda_input` and `cuda_vec` are equally fast;
  `cuda_vec`'s 4× fewer instructions buy nothing here.
- **`cuda_output` is not suitable for 2-byte types.** Selecting the source pointer
  (`(i & 1) ? b : a`) and doing one full-warp load and store might fix it; that remains
  untested.
- **The PyTorch reference is the correctness oracle only.** The lesson for later problems:
  write all bytes of a 32-byte sector in one place, at one time; per-instruction
  coalescing alone is not enough.

## Validation

- Original CUDA/PyTorch study on RTX 5080: **617 passed**, 117 skipped before
  Triton was implemented. Current Triton validation is documented below.
- Compute Sanitizer memcheck: **0 errors** in 468 CUDA tests, including misaligned inputs
  that exercise `cuda_vec`'s fallback.

## Limitations

- One desktop GPU with unlocked clocks; buffers are reused between samples.
- The benchmark tables and figure include Triton; the historical Nsight analysis
  covers CUDA/PyTorch only.
- The `cuda_output` fix above is a hypothesis, not measured.

Historical raw JSON and Nsight reports are in `results/rtx5080/` (not tracked by
Git). New benchmark JSON is in `artifacts/perf_20261023/interleave_arrays/`.
See [benchmarking](benchmarking.md) for reproduction.

## Triton implementation follow-up (2026-10-23)

The explicit Triton backend is now implemented. Combined GPU validation for
problems 4 and 5: **1,373 passed, no skips**. Triton-only memcheck: **227 passed,
zero errors**. See [testing](testing.md#triton-validation-2026-10-23) for commands
and coverage. The same-day performance follow-up now includes three matched
all-backend runs in the tables and chart above.
