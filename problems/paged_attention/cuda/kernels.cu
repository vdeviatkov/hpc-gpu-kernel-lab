// 49 · Paged Attention: CUDA kernels. Not implemented yet.
// Fill in paged_attention_kernel and launch_paged_attention, then set IMPLEMENTED = True in
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
__global__ void
paged_attention_kernel(const T *__restrict__ q, const T *__restrict__ k_cache,
                       const T *__restrict__ v_cache, const int32_t *__restrict__ block_table,
                       const int32_t *__restrict__ context_lens, T *__restrict__ out, int64_t b,
                       int64_t h, int64_t d, int64_t page_size, int64_t max_blocks) {
  // TODO: implement.
}

void launch_paged_attention(const at::Tensor &q, const at::Tensor &k_cache,
                            const at::Tensor &v_cache, const at::Tensor &block_table,
                            const at::Tensor &context_lens, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const int64_t b = q.size(0);
  const int64_t h = q.size(1);
  const int64_t d = q.size(2);
  const int64_t page_size = k_cache.size(2);
  const int64_t max_blocks = block_table.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch paged_attention_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)b, (void)h, (void)d, (void)page_size, (void)max_blocks;
  TORCH_CHECK(false, "paged_attention: CUDA kernel not implemented");
}
