// 22 · Sparse Matrix-Vector Multiplication: CUDA kernels. Not implemented yet.
// Fill in spmv_kernel and launch_spmv, then set IMPLEMENTED = True in implementation.py.
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
__global__ void spmv_kernel(const int32_t *__restrict__ row_ptr,
                            const int32_t *__restrict__ col_idx, const T *__restrict__ values,
                            const T *__restrict__ x, T *__restrict__ out, int64_t m, int64_t k,
                            int64_t nnz) {
  // TODO: implement.
}

void launch_spmv(const at::Tensor &row_ptr, const at::Tensor &col_idx, const at::Tensor &values,
                 const at::Tensor &x, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(row_ptr.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(row_ptr.get_device()).stream();
  const int64_t m = row_ptr.numel() - 1;
  const int64_t k = x.numel();
  const int64_t nnz = values.numel();
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch spmv_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)k, (void)nnz;
  TORCH_CHECK(false, "spmv: CUDA kernel not implemented");
}
