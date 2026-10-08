// 33 · RMS Normalization: CUDA kernels. Not implemented yet.
// Fill in rms_norm_kernel and launch_rms_norm, then set IMPLEMENTED = True in implementation.py.
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

template <typename T>
__global__ void rms_norm_kernel(const T *__restrict__ x, const T *__restrict__ weight, double eps,
                                T *__restrict__ out, int64_t m, int64_t n) {
  // TODO: implement.
}

void launch_rms_norm(const at::Tensor &x, const at::Tensor &weight, double eps, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t m = x.size(0);
  const int64_t n = x.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch rms_norm_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)n;
  TORCH_CHECK(false, "rms_norm: CUDA kernel not implemented");
}
