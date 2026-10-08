// 13 · Reduction: CUDA kernels. Not implemented yet.
// Fill in reduce_sum_kernel and launch_reduce_sum, then set IMPLEMENTED = True in
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
__global__ void reduce_sum_kernel(const T *__restrict__ x, float *__restrict__ out, int64_t n) {
  // TODO: implement.
}

void launch_reduce_sum(const at::Tensor &x, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t n = x.numel();
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch reduce_sum_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)n;
  TORCH_CHECK(false, "reduce_sum: CUDA kernel not implemented");
}
