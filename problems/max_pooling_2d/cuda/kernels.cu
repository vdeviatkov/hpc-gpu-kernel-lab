// 11 · 2D Max Pooling: CUDA kernels. Not implemented yet.
// Fill in max_pool2d_kernel and launch_max_pool2d, then set IMPLEMENTED = True in
// implementation.py. See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename T>
__global__ void max_pool2d_kernel(const T *__restrict__ x, int64_t kernel_size, int64_t stride,
                                  int64_t padding, T *__restrict__ out, int64_t n, int64_t c,
                                  int64_t h, int64_t w) {
  // TODO: implement.
}

void launch_max_pool2d(const at::Tensor &x, int64_t kernel_size, int64_t stride, int64_t padding,
                       at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t n = x.size(0);
  const int64_t c = x.size(1);
  const int64_t h = x.size(2);
  const int64_t w = x.size(3);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch max_pool2d_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)n, (void)c, (void)h, (void)w;
  TORCH_CHECK(false, "max_pool2d: CUDA kernel not implemented");
}
