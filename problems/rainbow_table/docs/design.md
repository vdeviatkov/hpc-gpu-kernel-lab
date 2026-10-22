# Design

[← Rainbow Table](../README.md) · [Results](results.md) · [Testing](testing.md) ·
[Benchmarking](benchmarking.md) · [Profiling](profiling.md) · [Roofline](roofline.md)

## Contract

For each int32 input, hash its four bytes least significant first with FNV-1a:
`h = (h ^ byte) * 16777619 mod 2^32`, starting at `2166136261`. Repeat on the
previous result R times. Output a new uint32 tensor. Source bounds are N ≤ 10M,
1 ≤ R ≤ 100; its performance case is N = 5M, R = 10. Empty inputs, R = 0,
R > 100 and negative bit patterns are lab extensions. Equality is bit exact.

## Variants and hypothesis

| Backend | Mechanism / question |
|---|---|
| pytorch | Source-style int64 arithmetic, explicit masks, many launches |
| pytorch_int32 | Wrap in int32; does narrower eager arithmetic help? |
| pytorch_compile | Fuse all rounds; compare a generated kernel |
| cuda / cuda_u32 | One uint32 chain per thread; these are aliases |
| cuda_int64 | Source arithmetic expressed in int64; inspect what survives optimization |
| cuda_ilp2 / cuda_ilp | Interleave 2 / 4 independent chains with coalesced scalar accesses |
| cuda_vec | Four chains and aligned 16-byte loads/stores; scalar ILP fallback for misalignment |
| cuda_unroll | Specialize R = 0,1,2,3,5,10,20,50,100; loop for other R |
| triton | Tiled uint32 hash, runtime round loop |
| triton_unroll | Same hash with static rounds |

CUDA defaults to 256 threads (128/256/512 supported). Triton defaults to 256
items and 4 warps, giving two elements per lane. Independent chains let a warp
issue work while another chain waits for its multiplication result. More resident
warps can hide that latency too; achieved occupancy and eligible warps distinguish
the mechanisms. Register count alone cannot decide the winner.

## Work and traffic

Logical traffic is 8N bytes, independent of R. Work is NR hash rounds; each round
has four XORs and four multiplies, hence 8NR **algorithmic integer operations**.
This excludes byte extraction and loop control and is not a SASS instruction count.
The generic benchmark's `flops` / `tflop_s` fields store these integer work counts;
divide `tflop_s` by eight to obtain trillion rounds/s. They are not floating-point rates.

A chain is serial across its four multiplies and across rounds. The three upper
bytes can be extracted early, but subsequent hash updates depend on the preceding
multiply. Read [SASS and the compute model](sass.md) before predicting throughput.
