// 37 · Categorical Cross Entropy Loss: CUDA kernels. Not implemented yet.
// Fill in cross_entropy_kernel and launch_cross_entropy, then set IMPLEMENTED = True in
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
__global__ void cross_entropy_kernel(const T *__restrict__ logits,
                                     const int64_t *__restrict__ labels, float *__restrict__ out,
                                     int64_t m, int64_t c) {
  // TODO: implement.
}

void launch_cross_entropy(const at::Tensor &logits, const at::Tensor &labels, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(logits.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  const int64_t m = logits.size(0);
  const int64_t c = logits.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch cross_entropy_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)c;
  TORCH_CHECK(false, "cross_entropy: CUDA kernel not implemented");
}
