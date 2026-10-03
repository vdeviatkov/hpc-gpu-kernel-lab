// 32 · Softmax: CUDA kernels. Not implemented yet.
// Fill in softmax_kernel and launch_softmax, then set IMPLEMENTED = True in implementation.py.
// See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename T>
__global__ void softmax_kernel(const T *__restrict__ x, T *__restrict__ out, int64_t m, int64_t n) {
  // TODO: implement.
}

void launch_softmax(const at::Tensor &x, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t m = x.size(0);
  const int64_t n = x.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch softmax_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)n;
  TORCH_CHECK(false, "softmax: CUDA kernel not implemented");
}
