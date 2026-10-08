// 25 · General Matrix Multiplication (GEMM) — FP16: CUDA kernels. Not implemented yet.
// Fill in gemm_kernel and launch_gemm, then set IMPLEMENTED = True in implementation.py.
// See problems/vector_add/cuda/kernels.cu for a complete example. Shared helpers:
// lab/vector.cuh (Pack, load_pack, store_pack, pack_aligned) and lab/launch.cuh
// (global_index, grid_threads, grid_blocks).
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lab/launch.cuh"
#include "lab/vector.cuh"

template <typename T>
__global__ void gemm_kernel(const T *__restrict__ a, const T *__restrict__ b,
                            const T *__restrict__ c, double alpha, double beta, T *__restrict__ out,
                            int64_t m, int64_t k, int64_t n) {
  // TODO: implement.
}

void launch_gemm(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c, double alpha,
                 double beta, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(a.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch gemm_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)k, (void)n;
  TORCH_CHECK(false, "gemm: CUDA kernel not implemented");
}
