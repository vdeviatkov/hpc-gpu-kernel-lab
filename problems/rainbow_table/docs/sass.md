# SASS and the compute model

[← Rainbow Table](../README.md) · [Roofline](roofline.md) · [Profiling](profiling.md)

Measured CUDA 13.4.92 toolkit output on this host targets `sm_120`; see captured
metadata for the exact nvcc version. Raw disassembly is in
`artifacts/rainbow_table/sass/cuda.sass`. Reproduce with:

```bash
cuobjdump --dump-sass artifacts/torch_extensions/gpu_lab_rainbow_table/*.so > cuda.sass
cuobjdump --dump-resource-usage artifacts/torch_extensions/gpu_lab_rainbow_table/*.so
python -m problems.rainbow_table.benchmarks.inspect_sass cuda.sass
```

## What actually executes

All CUDA variants multiply by `0x1000193` with **IMAD ..., RZ**: the addend is zero.
There is no IMUL and no PRMT. Byte extraction uses SHF right shifts at 8,16,24 bits.
The top-byte shift already produces an 8-bit result, so its XOR needs no mask.
`LOP3` implements Boolean truth tables, including `h ^ (shifted & 255)` in a single
instruction (`0x78` with the corresponding operands). The initial XOR with the seed
also fuses with masking (`0x6c`) in looped kernels. The unrolled CUDA version uses a
separate low-byte AND and seed XOR, costing one extra LOP3 per round.

Do not count every IMAD in a function as a hash multiply: IMAD.WIDE and IMAD.SHL
outside the round body calculate addresses. Likewise HFMA2 here can materialize
constants; it does not mean the algorithm performs half-precision arithmetic.

| CUDA variant | Hash body per thread per loop iteration | Other repeated instructions | Total / element / round | Registers |
|---|---|---|---:|---:|
| u32 (and cuda alias) | 4 IMAD + 4 LOP3 + 3 SHF | 1 HFMA2 constant + 2 uniform control + branch | 15 | 10 |
| int64 | 4 IMAD + 4 LOP3 + 3 SHF.R.U64 | 2 MOV + 2 uniform control + branch | 16 | 10 |
| ilp2 | 8 IMAD + 8 LOP3 + 6 SHF | 1 HFMA2 + 1 MOV + 2 uniform control + branch | 27 / 2 = 13.5 | 18 |
| ilp / vec | 16 IMAD + 16 LOP3 + 12 SHF | 2 HFMA2 + 2 MOV + 2 uniform control + branch | 51 / 4 = 12.75 | 30 / 20 |
| unroll, R ≥ 1 | per round: 4 IMAD + 5 LOP3 + 3 SHF | no round-loop control | 12 | 10 |

Uniform control means UIADD3.64 and UISETP. Counted totals are static issue
instructions along the round body, including uniform instructions; they are not
all on one execution pipe and must not be equated with a single pipeline's work.
All CUDA kernels have zero stack, shared-memory allocation and local memory in
cuobjdump resource usage. R = 0 unroll uses 8 registers.

## The int64 surprise

The compiler proves that masked products only need their low 32 bits and emits
one ordinary IMAD per multiply. There is **no wide-multiply carry sequence in the
hash body**. SHF.R.U64 and zero setup remain for extracting from an int64 value;
pointer/index arithmetic and the loop counter are also 64-bit, including in u32.
Do not label every `.64` instruction as expensive hash arithmetic. The int64 loop
backedge skips its initial zero setup: its repeated body is 0x130–0x220 (16
instructions), while u32 is 0x120–0x200 (15).

## From instructions to a ceiling

For K chains per thread, a full warp computes 32K rounds per loop iteration.
A dynamic warp-instruction count must therefore be normalized as
`32 * executed_warp_instructions / (N*R)` to compare with per-element work, with
extra care for inactive lanes, setup, and uniform instructions. For the hash math
alone, every CUDA variant needs four multiply issues per round and seven or eight
Boolean/shift issues. Model each pipe separately:

`Tround <= min(Pmultiply / 4, Pboolean_shift / (7 or 8), Pissue / Iround)`.

P values must be in lane instructions/s at measured clocks (or consistently in
warp issues/s). Add `8N / B` for memory and launch/setup time; finite occupancy and
dependency latency can lower the achievable rate further. Pipeline assignment
comes from the actual `sm__inst_executed_pipe_*` counters, not from the name IMAD.
The FP32 advertised TFLOP/s is not a valid integer hash ceiling. A measured large-R
plateau is an empirical ceiling for this kernel/configuration, not proof of peak
integer hardware throughput.

## Triton and compiled PyTorch

The Source/SASS exports provide a useful compiler contrast. Explicit looped
Triton handles two elements per lane. Each loop iteration has 8 IMAD, 8 LOP3,
4 PRMT, 2 SHF, 1 HFMA2, 1 MOV, 2 uniform loop controls and a branch: **27
instructions / 2 element-rounds = 13.5**. PRMT selectors `0x7771` and `0x7772`
select the middle bytes and fill the upper bytes with zero; the final byte uses
SHF by 24. XOR then needs no separate mask. This differs from CUDA's shifted
word plus fused AND/XOR.

Static Triton emits, per element-round, **4 IMAD + 5 LOP3 + 2 PRMT + 1 SHF = 12**.
At R=100 the two-chain kernel has 800 prime IMAD, 1000 hash LOP3, 400 PRMT and
200 right SHF, excluding address setup. Compiled PyTorch's Triton kernel has the
same hash counts. Check `triton_r100.sass`, `triton_unroll_r100.sass` and
`pytorch_compile_r100.sass` in the capture directory. The register table in
[results](results.md) uses runtime launch metrics: 16 for explicit Triton and
16 for compiled PyTorch at R=100 (compiled R=1 uses 26).

Runtime CUDA launch allocation for scalar/unroll is 16 registers/thread even
though cuobjdump reports 10 static registers. ILP2/ILP4/vec report 18/30/20 at
runtime. Keep static resource counts and runtime launch allocation distinct.
