# Results

[← Interleave Arrays](../README.md) · [Design](design.md) · **Results** ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

**Setup:** NVIDIA RTX 5080 (64 MiB L2, DRAM peak 960 GB/s), CUDA 13.2, PyTorch 2.14.1,
stock clocks, idle GPU. Each number is the median of 100 samples after 25 warmups, taken
as the median of three full runs (spread between runs ≤ 0.4% for every backend at
N ≥ 25M). Timing uses CUDA events around one call, including output allocation. Inputs
are normally distributed. Measured from commit `0e9f2ef` plus the CUDA kernels and the
`pytorch_stack` baseline. Commands are in [benchmarking](benchmarking.md).

## DRAM-bound sizes

Latency in µs, lower is better; percentage of copy speed in brackets. The copy reference
copies both inputs, the same bytes the kernels move.

| N = 25,000,000 | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| copy of both inputs (target) | 488.0 | 243.2 | 243.3 |
| `pytorch` (reference) | 1370.3 (36%) | 685.7 (35%) | 685.7 (35%) |
| `pytorch_stack` | 1353.2 (36%) | 687.5 (35%) | 686.6 (35%) |
| `cuda_input` | 493.9 (99%) | 248.3 (98%) | 249.0 (98%) |
| `cuda_output` | 494.1 (99%) | **403.0 (60%)** | **402.3 (61%)** |
| `cuda_pair_store` | 494.4 (99%) | **247.5 (98%)** | **247.5 (98%)** |
| `cuda_vec` | 495.9 (98%) | 250.4 (97%) | 250.4 (97%) |

At N = 50,000,000 (the contract maximum) the picture is identical: the FP32 kernels at
99% of a copy (975.4 µs), `cuda_output` FP16 at 61% (795.9 vs 486.9 µs), PyTorch at 36%.

- In FP32 all four CUDA kernels are within 1.6% of a copy: the mapping barely matters.
- In FP16/BF16 only `cuda_output` falls behind, by 40%.
- Both PyTorch baselines take 2.8× as long as a copy.

## Why the variants differ

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

At N = 1025 every backend is launch-bound: 10.4–10.7 µs for the CUDA kernels, 10.7–11.1 µs
for the reference, 7.8 µs for `pytorch_stack` (one kernel). The copy reference makes two
launches here (8.6–8.7 µs), so compare small sizes in µs rather than percent of copy. At
N = 1M FP16, `cuda_output` again trails (16.4 µs against 12.1–12.3 µs for the other CUDA
kernels).

## Decisions

- **`cuda_pair_store` is the recommended kernel:** at copy speed for every dtype, full
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

- Test suite on the RTX 5080: **617 passed**; 117 skipped are the deferred Triton backend
  ([testing](testing.md)).
- Compute Sanitizer memcheck: **0 errors** in 468 CUDA tests, including misaligned inputs
  that exercise `cuda_vec`'s fallback.

## Limitations

- One desktop GPU with unlocked clocks; buffers are reused between samples.
- Triton is deferred.
- The `cuda_output` fix above is a hypothesis, not measured.

Raw JSON and the Nsight reports are in `results/rtx5080/` (not tracked by Git);
[sweep.sh](../benchmarks/sweep.sh) regenerates both.
