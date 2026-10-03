// 07 · Matrix Copy: CUDA kernels. Not implemented yet.
// Fill in matrix_copy_kernel and launch_matrix_copy, then set IMPLEMENTED = True in
// implementation.py. See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename T>
__global__ void matrix_copy_kernel(const T *__restrict__ a, T *__restrict__ out, int64_t m,
                                   int64_t n) {
  // TODO: implement.
}

void launch_matrix_copy(const at::Tensor &a, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(a.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const int64_t m = a.size(0);
  const int64_t n = a.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch matrix_copy_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)n;
  TORCH_CHECK(false, "matrix_copy: CUDA kernel not implemented");
}
