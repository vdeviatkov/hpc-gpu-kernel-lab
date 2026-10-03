// 29 · Fused GEMM + Bias + Activation: CUDA kernels. Not implemented yet.
// Fill in linear_relu_kernel and launch_linear_relu, then set IMPLEMENTED = True in
// implementation.py. See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename T>
__global__ void linear_relu_kernel(const T *__restrict__ x, const T *__restrict__ w,
                                   const T *__restrict__ bias, T *__restrict__ out, int64_t m,
                                   int64_t k, int64_t n) {
  // TODO: implement.
}

void launch_linear_relu(const at::Tensor &x, const at::Tensor &w, const at::Tensor &bias,
                        at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = w.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch linear_relu_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)k, (void)n;
  TORCH_CHECK(false, "linear_relu: CUDA kernel not implemented");
}
