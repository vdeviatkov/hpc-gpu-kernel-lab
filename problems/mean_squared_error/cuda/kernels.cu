// 15 · Mean Squared Error: CUDA kernels. Not implemented yet.
// Fill in mse_kernel and launch_mse, then set IMPLEMENTED = True in implementation.py.
// See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename T>
__global__ void mse_kernel(const T *__restrict__ pred, const T *__restrict__ target,
                           float *__restrict__ out, int64_t n) {
  // TODO: implement.
}

void launch_mse(const at::Tensor &pred, const at::Tensor &target, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(pred.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(pred.get_device()).stream();
  const int64_t n = pred.numel();
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch mse_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)n;
  TORCH_CHECK(false, "mse: CUDA kernel not implemented");
}
