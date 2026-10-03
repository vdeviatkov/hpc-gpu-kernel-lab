// 50 · MoE Token Routing and Dispatch: CUDA kernels. Not implemented yet.
// Fill in route_tokens_kernel and launch_route_tokens, then set IMPLEMENTED = True in
// implementation.py. See problems/vector_add/cuda/kernels.cu for a complete example.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename T>
__global__ void
route_tokens_kernel(const T *__restrict__ x, const float *__restrict__ router_logits, int64_t top_k,
                    T *__restrict__ dispatched, int32_t *__restrict__ expert_offsets,
                    int32_t *__restrict__ token_index, float *__restrict__ weights, int64_t t,
                    int64_t d, int64_t e) {
  // TODO: implement.
}

void launch_route_tokens(const at::Tensor &x, const at::Tensor &router_logits, int64_t top_k,
                         at::Tensor dispatched, at::Tensor expert_offsets, at::Tensor token_index,
                         at::Tensor weights) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t t = x.size(0);
  const int64_t d = x.size(1);
  const int64_t e = router_logits.size(1);
  // TODO: choose a grid, dispatch on dtype (AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,
  // at::kBFloat16, ...)), and launch route_tokens_kernel on `stream`, then call
  // C10_CUDA_KERNEL_LAUNCH_CHECK().
  (void)stream;
  (void)t, (void)d, (void)e;
  TORCH_CHECK(false, "route_tokens: CUDA kernel not implemented");
}
