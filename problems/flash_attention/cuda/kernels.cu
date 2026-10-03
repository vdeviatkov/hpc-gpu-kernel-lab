// 48 · FlashAttention-Style Online Attention: CUDA kernels. Not implemented yet.
// Fill in flash_attention_kernel and launch_flash_attention, then set IMPLEMENTED = True in
// implementation.py. See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename T>
__global__ void flash_attention_kernel(const T *__restrict__ q, const T *__restrict__ k,
                                       const T *__restrict__ v, bool causal, T *__restrict__ out,
                                       int64_t b, int64_t h, int64_t s, int64_t d) {
  // TODO: implement.
}

void launch_flash_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                            bool causal, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const int64_t b = q.size(0);
  const int64_t h = q.size(1);
  const int64_t s = q.size(2);
  const int64_t d = q.size(3);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch flash_attention_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)b, (void)h, (void)s, (void)d;
  TORCH_CHECK(false, "flash_attention: CUDA kernel not implemented");
}
