// 05 · RGB to Grayscale: CUDA kernels. Not implemented yet.
// Fill in rgb_to_grayscale_kernel and launch_rgb_to_grayscale, then set IMPLEMENTED = True in
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
__global__ void rgb_to_grayscale_kernel(const T *__restrict__ image, T *__restrict__ out, int64_t h,
                                        int64_t w) {
  // TODO: implement.
}

void launch_rgb_to_grayscale(const at::Tensor &image, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(image.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(image.get_device()).stream();
  const int64_t h = image.size(0);
  const int64_t w = image.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch rgb_to_grayscale_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)h, (void)w;
  TORCH_CHECK(false, "rgb_to_grayscale: CUDA kernel not implemented");
}
