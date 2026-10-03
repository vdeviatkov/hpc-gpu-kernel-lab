// 12 · Weight Dequantization: CUDA kernels. Not implemented yet.
// Fill in dequantize_kernel and launch_dequantize, then set IMPLEMENTED = True in
// implementation.py. See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename T>
__global__ void dequantize_kernel(const int8_t *__restrict__ q, const T *__restrict__ scale,
                                  int64_t quant_block, T *__restrict__ out, int64_t m, int64_t n) {
  // TODO: implement.
}

void launch_dequantize(const at::Tensor &q, const at::Tensor &scale, int64_t quant_block,
                       at::Tensor out) {
  const c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const int64_t m = q.size(0);
  const int64_t n = q.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch dequantize_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)n;
  TORCH_CHECK(false, "dequantize: CUDA kernel not implemented");
}
