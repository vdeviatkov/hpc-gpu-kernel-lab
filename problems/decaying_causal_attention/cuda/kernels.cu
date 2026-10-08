// 45 · Decaying Causal Attention: CUDA kernels. Not implemented yet.
// Fill in decaying_attention_kernel and launch_decaying_attention, then set IMPLEMENTED = True in
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
__global__ void decaying_attention_kernel(const T *__restrict__ q, const T *__restrict__ k,
                                          const T *__restrict__ v, double gamma,
                                          T *__restrict__ out, int64_t s, int64_t d) {
  // TODO: implement.
}

void launch_decaying_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                               double gamma, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const int64_t s = q.size(0);
  const int64_t d = q.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch decaying_attention_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)s, (void)d;
  TORCH_CHECK(false, "decaying_attention: CUDA kernel not implemented");
}
