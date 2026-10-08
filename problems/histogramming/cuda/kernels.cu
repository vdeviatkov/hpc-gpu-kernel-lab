// 18 · Histogramming: CUDA kernels. Not implemented yet.
// Fill in histogram_kernel and launch_histogram, then set IMPLEMENTED = True in implementation.py.
// See problems/vector_add/cuda/kernels.cu for a complete example. Shared helpers:
// lab/vector.cuh (Pack, load_pack, store_pack, pack_aligned) and lab/launch.cuh
// (global_index, grid_threads, grid_blocks).
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lab/launch.cuh"
#include "lab/vector.cuh"

__global__ void histogram_kernel(const int32_t *__restrict__ x, int64_t bins,
                                 int32_t *__restrict__ out, int64_t n) {
  // TODO: implement.
}

void launch_histogram(const at::Tensor &x, int64_t bins, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t n = x.numel();
  // TODO: choose a grid, launch,
  // and launch histogram_kernel on `stream`, then call C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)n;
  TORCH_CHECK(false, "histogram: CUDA kernel not implemented");
}
