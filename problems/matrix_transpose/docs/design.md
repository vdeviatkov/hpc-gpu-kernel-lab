[← Matrix Transpose](../README.md) · **Design** · [Results](results.md) ·
[Benchmarking](benchmarking.md) · [Profiling](profiling.md)

# Matrix Transpose: steps 0–3

## 0. Source verification

Verified the live [LeetGPU challenge](https://leetgpu.com/challenges/matrix-transpose)
on 2026-10-24. It specifies row-major fp32 input and output, dimensions from 1
through 8192 inclusive, and a performance case of rows=7000, cols=6000. The two
worked shapes are 2×3 and 3×1. The result is written to the supplied output in
the source interface. The displayed JAX starter accepts `(input, rows, cols)`
and returns the output; this lab uses an allocating tensor API instead.
No challenge submission was made. Checker tolerance and hidden tests were not
available from the description; the lab requires exact element bits.

## 1. Algorithm and contract

For input A of shape M×N, set B[j,i] = A[i,j]. Flat offsets are `i*N+j`
and `j*M+i`. This is a permutation, with no arithmetic, accumulation or
precision conversion. Each destination has exactly one source and one writer.

The API accepts contiguous strided 2-D fp32/fp16/bf16 tensors and returns a fresh
contiguous N×M tensor of the same dtype and device. Input storage offsets are
allowed; the input is unchanged. Strided/pitched inputs are rejected. Empty
matrices, dimensions above 8192, and fp16/bf16 are labeled lab extensions.
NaNs (including payloads), infinities, signed zeros and subnormals are moved
without arithmetic. An empty input produces the reversed shape with no work.

## 2. Correctness reference

`a.t().clone(memory_format=torch.contiguous_format)` materializes the result.
A transposed view alone does not satisfy the output layout or ownership contract.
Even `a.t().contiguous()` can alias a singleton input, so cloning is required.
CPU tests independently construct expected outputs by indexed assignment, check
ownership and mutation, cover rectangular/tile-tail/empty/offset inputs, and
compare integer views for special bit patterns. CUDA/Triton remain scaffold stubs.

## 3. Simple performance model

For E=M*N elements of s bytes each:

- Useful traffic: `B=2*E*s` bytes (one read and one write).
- Useful floating-point work: zero; report GB/s, with no TFLOP/s value.
- Output storage: `E*s`; the reference needs no full-size intermediate beyond it.
- Independent work: E element transfers, O(E) work. A naive kernel can assign one
  input element per thread; address generation is integer work.
- At bandwidth W bytes/s, the optimistic bandwidth bound is `B/W` seconds.
  Include launch latency L as a separate small-input floor: `max(L, B/W)`.
  This omits allocation, transaction amplification, address instructions and
  resource/synchronization costs; it is not a latency prediction.

The source fp32 case has 42,000,000 elements, 168,000,000 output bytes, and
336,000,000 useful traffic bytes. At a hypothetical 1 TB/s, the traffic bound is
336 microseconds (no device bandwidth is assumed or measured). fp16/bf16 halve
those byte counts. At 1024² fp32, useful traffic is 8,388,608 bytes.

Consecutive threads reading a row obtain contiguous loads but write addresses
M elements apart. Reversing ownership coalesces stores and scatters loads.
A future shared tile can exchange ownership so both global directions coalesce;
it adds shared writes/reads and a block barrier. For a 32×32 fp32 shared tile
under a 32-bank, 4-byte bank model, column reads map lanes to the same bank;
a 32×33 padded tile changes the bank mapping. These are hypotheses for later
GPU implementation and profiling, not measured findings, and the fp16/bf16
bank behavior needs separate analysis.

Compare measured transpose against problem 7's copy on identical shape, dtype,
cache state and timing scope. Logical bytes are not measured DRAM traffic;
cache residency and scattered transactions can change the relevant bandwidth.
Measured on an RTX 5080: the bank-conflict hypotheses hold exactly; see [results](results.md).
