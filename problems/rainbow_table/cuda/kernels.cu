// 06 · Rainbow Table: CUDA kernels. Not implemented yet.
// Fill in rainbow_table_kernel and launch_rainbow_table, then set IMPLEMENTED = True in
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

__global__ void rainbow_table_kernel(const int32_t *__restrict__ x, int64_t rounds,
                                     int32_t *__restrict__ out, int64_t n) {
  // TODO: implement.
}

void launch_rainbow_table(const at::Tensor &x, int64_t rounds, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t n = x.numel();
  // TODO: choose a grid, launch,
  // and launch rainbow_table_kernel on `stream`, then call C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)n;
  TORCH_CHECK(false, "rainbow_table: CUDA kernel not implemented");
}
