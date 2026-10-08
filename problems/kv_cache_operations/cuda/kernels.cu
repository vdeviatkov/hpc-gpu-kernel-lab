// 47 · KV-Cache Update and Paged Access: CUDA kernels. Not implemented yet.
// Fill in kv_cache_append_kernel and launch_kv_cache_append, then set IMPLEMENTED = True in
// implementation.py. See problems/vector_add/cuda/kernels.cu for a complete example.
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
__global__ void kv_cache_append_kernel(const T *__restrict__ k_new, const T *__restrict__ v_new,
                                       T *__restrict__ k_cache, T *__restrict__ v_cache,
                                       const int32_t *__restrict__ block_table,
                                       const int32_t *__restrict__ positions, int64_t b, int64_t h,
                                       int64_t t, int64_t d, int64_t page_size) {
  // TODO: implement.
}

void launch_kv_cache_append(const at::Tensor &k_new, const at::Tensor &v_new, at::Tensor k_cache,
                            at::Tensor v_cache, const at::Tensor &block_table,
                            const at::Tensor &positions) {
  const c10::cuda::CUDAGuard guard(k_new.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(k_new.get_device()).stream();
  const int64_t b = k_new.size(0);
  const int64_t h = k_new.size(1);
  const int64_t t = k_new.size(2);
  const int64_t d = k_new.size(3);
  const int64_t page_size = k_cache.size(2);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch kv_cache_append_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)b, (void)h, (void)t, (void)d, (void)page_size;
  TORCH_CHECK(false, "kv_cache_append: CUDA kernel not implemented");
}
