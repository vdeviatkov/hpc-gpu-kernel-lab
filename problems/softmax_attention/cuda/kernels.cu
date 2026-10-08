// 40 · Softmax Attention: CUDA kernels. Not implemented yet.
// Fill in attention_kernel and launch_attention, then set IMPLEMENTED = True in implementation.py.
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
__global__ void attention_kernel(const T *__restrict__ q, const T *__restrict__ k,
                                 const T *__restrict__ v, T *__restrict__ out, int64_t m, int64_t n,
                                 int64_t d) {
  // TODO: implement.
}

void launch_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                      at::Tensor out) {
  const c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const int64_t m = q.size(0);
  const int64_t n = k.size(0);
  const int64_t d = q.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch attention_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)n, (void)d;
  TORCH_CHECK(false, "attention: CUDA kernel not implemented");
}
