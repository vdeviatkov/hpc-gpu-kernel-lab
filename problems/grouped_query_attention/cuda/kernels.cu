// 44 · Grouped Query Attention: CUDA kernels. Not implemented yet.
// Fill in gqa_kernel and launch_gqa, then set IMPLEMENTED = True in implementation.py.
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
__global__ void gqa_kernel(const T *__restrict__ q, const T *__restrict__ k,
                           const T *__restrict__ v, T *__restrict__ out, int64_t b, int64_t hq,
                           int64_t hkv, int64_t s, int64_t d) {
  // TODO: implement.
}

void launch_gqa(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const int64_t b = q.size(0);
  const int64_t hq = q.size(1);
  const int64_t hkv = k.size(1);
  const int64_t s = q.size(2);
  const int64_t d = q.size(3);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch gqa_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)b, (void)hq, (void)hkv, (void)s, (void)d;
  TORCH_CHECK(false, "gqa: CUDA kernel not implemented");
}
