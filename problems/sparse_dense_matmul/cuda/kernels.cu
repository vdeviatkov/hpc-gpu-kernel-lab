// 28 · Sparse Matrix-Dense Matrix Multiplication: CUDA kernels. Not implemented yet.
// Fill in spmm_kernel and launch_spmm, then set IMPLEMENTED = True in implementation.py.
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
__global__ void spmm_kernel(const int32_t *__restrict__ row_ptr,
                            const int32_t *__restrict__ col_idx, const T *__restrict__ values,
                            const T *__restrict__ b, T *__restrict__ out, int64_t m, int64_t k,
                            int64_t n, int64_t nnz) {
  // TODO: implement.
}

void launch_spmm(const at::Tensor &row_ptr, const at::Tensor &col_idx, const at::Tensor &values,
                 const at::Tensor &b, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(row_ptr.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(row_ptr.get_device()).stream();
  const int64_t m = row_ptr.numel() - 1;
  const int64_t k = b.size(0);
  const int64_t n = b.size(1);
  const int64_t nnz = values.numel();
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch spmm_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)k, (void)n, (void)nnz;
  TORCH_CHECK(false, "spmm: CUDA kernel not implemented");
}
