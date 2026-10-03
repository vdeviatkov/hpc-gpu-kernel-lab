// 27 · INT8 Quantized MatMul: CUDA kernels. Not implemented yet.
// Fill in int8_matmul_kernel and launch_int8_matmul, then set IMPLEMENTED = True in
// implementation.py. See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

__global__ void int8_matmul_kernel(const int8_t *__restrict__ a, const int8_t *__restrict__ b,
                                   double scale_a, double scale_b, int64_t zero_a, int64_t zero_b,
                                   float *__restrict__ out, int64_t m, int64_t k, int64_t n) {
  // TODO: implement.
}

void launch_int8_matmul(const at::Tensor &a, const at::Tensor &b, double scale_a, double scale_b,
                        int64_t zero_a, int64_t zero_b, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(a.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);
  // TODO: choose a grid, launch,
  // and launch int8_matmul_kernel on `stream`, then call C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)m, (void)k, (void)n;
  TORCH_CHECK(false, "int8_matmul: CUDA kernel not implemented");
}
