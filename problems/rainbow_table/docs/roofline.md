# Integer roofline

[← Rainbow Table](../README.md) · [Results](results.md) · [SASS](sass.md)

Use hash rounds as the work unit. Logical intensity is
`I = NR / (8N) = R/8` rounds/byte. With sustainable bandwidth B bytes/s and a
compute ceiling C rounds/s, the ideal throughput is `min(B*R/8, C)` and the ridge
is at `R = 8*C/B`. Doubling R initially doubles rounds/s at the same latency;
after the ridge, latency grows with R and rounds/s approaches C. An additive
latency model `time ≈ fixed + 8N/B + NR/C` gives a smooth transition; overlap can
make the maximum of memory and compute terms a better lower bound.

The [headline chart](../figures/r_sweep.svg) uses N = 10M (80 MB input + output),
and marks a 960 GB/s nominal DRAM line. It is a logical-traffic reference, not a
measured sustainable roof. The benchmark reuses inputs without cache flushing;
the source N = 5M has only 40 MB logical traffic and fits in this GPU's L2. Even
80 MB can have partial cache reuse. Compare with the copy control and DRAM/L2
counters before describing R = 1 as DRAM-bound. Dependency-limited variants may
flatten below the hardware compute ceiling; eager backends pay many launches and
intermediate-memory passes, so 8N is not their physical traffic.

## Why the standard Nsight roofline is insufficient

Nsight's default scalar rooflines count floating-point work; this hash has none.
A missing or near-zero FP point cannot demonstrate a move across an integer ridge.
Use the custom rounds/byte chart, Speed of Light memory vs SM utilization, integer
pipe counters, SASS, and scheduler statistics together. For a physical DRAM view,
use `NR / (dram_read_bytes + dram_write_bytes)` and `NR / profiler_seconds` from
the **same capture**. Writes retained in L2 make this a per-kernel counter view,
not necessarily the eventual complete traffic. For an L2 roof use L2 bytes and
an L2 bandwidth ceiling consistently. Never combine logical intensity with a
measured physical-traffic ceiling without explaining the distinction.

NVIDIA describes the roofline sections and throughput normalization in its
[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
