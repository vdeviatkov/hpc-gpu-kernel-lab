// 39 · SwiGLU MLP Block: CUDA kernels. Not implemented yet.
// Fill in swiglu_mlp_kernel and launch_swiglu_mlp, then set IMPLEMENTED = True in
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
__global__ void swiglu_mlp_kernel(const T *__restrict__ x, const T *__restrict__ w_gate,
                                  const T *__restrict__ w_up, const T *__restrict__ w_down,
                                  T *__restrict__ out, int64_t m, int64_t d, int64_t f) {
  // TODO: implement.
}

void launch_swiglu_mlp(const at::Tensor &x, const at::Tensor &w_gate, const at::Tensor &w_up,
                       const at::Tensor &w_down, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t m = x.size(0);
  const int64_t d = x.size(1);
  const int64_t f = w_gate.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch swiglu_mlp_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)d, (void)f;
  TORCH_CHECK(false, "swiglu_mlp: CUDA kernel not implemented");
}
