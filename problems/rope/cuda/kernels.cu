// 38 · Rotary Positional Embedding: CUDA kernels. Not implemented yet.
// Fill in rope_kernel and launch_rope, then set IMPLEMENTED = True in implementation.py.
// See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename T>
__global__ void rope_kernel(const T *__restrict__ x, const T *__restrict__ cos,
                            const T *__restrict__ sin, T *__restrict__ out, int64_t b, int64_t s,
                            int64_t h, int64_t d) {
  // TODO: implement.
}

void launch_rope(const at::Tensor &x, const at::Tensor &cos, const at::Tensor &sin,
                 at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t b = x.size(0);
  const int64_t s = x.size(1);
  const int64_t h = x.size(2);
  const int64_t d = x.size(3);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch rope_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)b, (void)s, (void)h, (void)d;
  TORCH_CHECK(false, "rope: CUDA kernel not implemented");
}
