// 23 · Matrix Multiplication: CUDA kernels. Not implemented yet.
// Fill in matmul_kernel and launch_matmul, then set IMPLEMENTED = True in implementation.py.
// See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename T>
__global__ void matmul_kernel(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out,
                              int64_t m, int64_t k, int64_t n) {
  // TODO: implement.
}

void launch_matmul(const at::Tensor &a, const at::Tensor &b, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(a.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch matmul_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)k, (void)n;
  TORCH_CHECK(false, "matmul: CUDA kernel not implemented");
}
