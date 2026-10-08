// 35 · Batch Normalization: CUDA kernels. Not implemented yet.
// Fill in batch_norm_kernel and launch_batch_norm, then set IMPLEMENTED = True in
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
__global__ void batch_norm_kernel(const T *__restrict__ x, const T *__restrict__ weight,
                                  const T *__restrict__ bias, double eps, T *__restrict__ out,
                                  int64_t n, int64_t c, int64_t hw) {
  // TODO: implement.
}

void launch_batch_norm(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &bias,
                       double eps, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t n = x.size(0);
  const int64_t c = x.size(1);
  const int64_t hw = x.size(2) * x.size(3);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch batch_norm_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)n, (void)c, (void)hw;
  TORCH_CHECK(false, "batch_norm: CUDA kernel not implemented");
}
