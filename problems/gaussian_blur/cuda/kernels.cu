// 10 · Gaussian Blur: CUDA kernels. Not implemented yet.
// Fill in gaussian_blur_kernel and launch_gaussian_blur, then set IMPLEMENTED = True in
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
__global__ void gaussian_blur_kernel(const T *__restrict__ image, const T *__restrict__ kernel,
                                     T *__restrict__ out, int64_t h, int64_t w, int64_t k) {
  // TODO: implement.
}

void launch_gaussian_blur(const at::Tensor &image, const at::Tensor &kernel, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(image.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(image.get_device()).stream();
  const int64_t h = image.size(0);
  const int64_t w = image.size(1);
  const int64_t k = kernel.size(0);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch gaussian_blur_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)h, (void)w, (void)k;
  TORCH_CHECK(false, "gaussian_blur: CUDA kernel not implemented");
}
