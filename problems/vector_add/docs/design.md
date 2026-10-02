# Design

[← Vector Addition](../README.md) · **Design** · [Results](results.md) ·
[Testing](testing.md) · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

## Source and contract

[LeetGPU Vector Addition](https://leetgpu.com/challenges/vector-addition), Easy.
Inspected on 2026-09-12: equal-length FP32 vectors, `C[i] = A[i] + B[i]`,
1 ≤ N ≤ 100,000,000; the performance case uses N = 25,000,000.
These are library-integrated experiments, not challenge submissions.

**Lab extensions:** empty vectors, FP16/BF16, and contiguous storage-offset views.
The checked entry point is `api.add(a, b, backend=..., out=None)`:

- Equal contiguous 1-D tensors, matching dtype/device; no broadcasting.
- FP32 arithmetic, with FP16/BF16 outputs rounded after an FP32 intermediate.
- Inputs may alias each other; output must not overlap either input. Disjoint
  slices of the same storage are allowed. Inputs are never mutated.
- Fresh output unless `out` is supplied; empty inputs launch no device kernel.
- Forward-only: autograd and unresolved negative/conjugate views are rejected.
- CUDA/Triton use the input device and PyTorch's current stream, asynchronously.
  Consumers on other streams must establish their own dependencies.
- CPU PyTorch/JAX support correctness checks, not GPU performance claims.

## Implementations

The four CUDA variants form a 2×2 design, so each comparison changes one factor:

| | One item per thread | Capped grid-stride loop (≤ 4096 blocks) |
|---|---|---|
| **Scalar access** | `cuda_scalar` | `cuda_grid_stride` |
| **16-byte vector access** | `cuda_vec` | `cuda_vec_grid_stride` |

A 16-byte pack holds four FP32 or eight FP16/BF16 elements, loaded with
`LDG.E.128` and stored with `STG.E.128` (verified in SASS for all three dtypes).
Elements that do not fill a whole pack are handled in the same launch. If any of the
three pointers is not 16-byte aligned, the vector variant falls back along the vector
axis only (`cuda_vec` → `cuda_scalar`, `cuda_vec_grid_stride` → `cuda_grid_stride`),
so the loop structure is kept. All pointers are `__restrict__`; inputs therefore use
the read-only `.CONSTANT` load path.

| Other backends | Mechanism |
|---|---|
| `pytorch` | Native eager add, with optional output buffer |
| `triton` | Masked tiles; default 1024 elements and 4 warps; no autotuning |
| JAX | Native arrays with JIT compilation and explicit completion |

Source: [CUDA kernels](../cuda/kernels.cu), [bindings](../cuda/bindings.cpp),
[Triton](../triton/implementation.py), [PyTorch](../pytorch/implementation.py),
[JAX](../jax/implementation.py). All support FP32/FP16/BF16.

[api.py](../api.py) owns the public PyTorch-tensor contract; backend modules are
internal. JAX exposes a separate native-array interface. CUDA repeats validation
at the C++ boundary, uses an RAII device guard and checks launch errors without
inserting synchronization.

## Performance model

N additions move a minimum of `3*N*s` logical bytes for element size s: two
reads and one write. Arithmetic intensity is `1/(3*s)` FLOP/byte, or 1/12
for FP32. At N = 25,000,000, logical traffic is 300 MB for FP32.

At a bandwidth ceiling W bytes/s, `3*N*s/W` is an optimistic memory-time bound.
The RTX 5080's spec ceiling is 960 GB/s (16 GB GDDR7, 256-bit bus at 30 Gbps), so
25M FP32 elements need at least 312.5 µs. Effective GB/s is `3*N*s / seconds / 1e9`,
not measured DRAM traffic. There is no data reuse to justify shared memory and no
Tensor Core operation here.

Every benchmark case also times a device-to-device `Tensor.copy_` (`copy_reference`)
as a measured bandwidth reference. Copy is half writes, while addition is one third
writes, so copy is **not** a strict upper bound: for FP32 at 25M elements, addition
(846 GB/s) beats copy (817 GB/s). The spec peak is the ceiling; the copy reference
shows what a vendor-tuned streaming operation achieves on the same buffers.

## References

- [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Triton vector-add tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Lab methodology](../../../docs/methodology.md)
