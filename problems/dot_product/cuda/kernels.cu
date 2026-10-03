// 16 · Dot Product: CUDA kernels. Not implemented yet.
// Fill in dot_kernel and launch_dot, then set IMPLEMENTED = True in implementation.py.
// See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename T>
__global__ void dot_kernel(const T *__restrict__ a, const T *__restrict__ b,
                           float *__restrict__ out, int64_t n) {
  // TODO: implement.
}

void launch_dot(const at::Tensor &a, const at::Tensor &b, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(a.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const int64_t n = a.numel();
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch dot_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)n;
  TORCH_CHECK(false, "dot: CUDA kernel not implemented");
}
