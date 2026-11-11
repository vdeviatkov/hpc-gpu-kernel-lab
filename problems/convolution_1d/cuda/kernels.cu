// 09 · 1D Convolution: CUDA kernels.
//
// out[i] = sum_j x[i + j] * w[j] for i < m = N - K + 1, accumulated in FP32 and rounded once
// to the input dtype. One thread per output in both variants:
//
//   0 conv1d_naive  x and w read from global memory. Per tap a warp issues one coalesced
//                   (but j-misaligned) x load and one broadcast w load; L1 absorbs the
//                   K-fold reuse of x, so the LSU/L1 pipe, not DRAM, is the expected limit.
//   1 conv1d_const  w converted to FP32 and staged in __constant__ memory. Every lane reads
//                   the same tap, which the constant cache broadcasts, so the w loads leave
//                   the LSU/L1 pipe; x loads are unchanged. Capacity: kMaxConstTaps taps.
//
// __constant__ memory is one buffer per device, shared by every call. Calls on one stream
// are ordered (each upload waits for the previous kernel), but conv1d_const calls running
// concurrently on different streams of the same device would overwrite each other's filter.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

#include "lab/launch.cuh"

namespace {

constexpr int kThreads = 256;
// 64 KiB, the whole user __constant__ bank: 16384 FP32 taps for every input dtype.
constexpr int64_t kMaxConstTaps = 16384;

__constant__ float c_w[kMaxConstTaps];

template <typename T>
__global__ void conv1d_naive(const T *__restrict__ x, const T *__restrict__ w, T *__restrict__ out,
                             int64_t m, int k) {
  const int64_t i = lab::global_index();
  if (i >= m)
    return;
  const T *xi = x + i;
  float acc = 0.0f;
  for (int j = 0; j < k; ++j)
    acc = fmaf(static_cast<float>(xi[j]), static_cast<float>(w[j]), acc);
  out[i] = static_cast<T>(acc);
}

template <typename T>
__global__ void conv1d_const(const T *__restrict__ x, T *__restrict__ out, int64_t m, int k) {
  const int64_t i = lab::global_index();
  if (i >= m)
    return;
  const T *xi = x + i;
  float acc = 0.0f;
  for (int j = 0; j < k; ++j)
    acc = fmaf(static_cast<float>(xi[j]), c_w[j], acc);
  out[i] = static_cast<T>(acc);
}

} // namespace

int64_t max_const_taps() { return kMaxConstTaps; }

void launch_conv1d(const at::Tensor &x, const at::Tensor &w, at::Tensor out, int64_t variant) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t n = x.numel();
  const int64_t k = w.numel();
  const int64_t m = n - k + 1;
  TORCH_CHECK(k <= INT_MAX, "conv1d: filter longer than INT_MAX taps");
  const int blocks = lab::grid_blocks(m, kThreads);

  if (variant == 1) {
    TORCH_CHECK(k <= kMaxConstTaps, "conv1d: cuda_const holds at most ", kMaxConstTaps,
                " taps in __constant__ memory");
    // Converted on the current stream (no copy for FP32); the caching allocator keeps the
    // temporary alive until the stream has consumed it.
    const at::Tensor w32 = w.to(at::kFloat);
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_w, w32.data_ptr<float>(), k * sizeof(float), 0,
                                           cudaMemcpyDeviceToDevice, stream));
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "conv1d", [&] {
    const auto *xp = x.data_ptr<scalar_t>();
    auto *op = out.data_ptr<scalar_t>();
    if (variant == 0)
      conv1d_naive<scalar_t><<<blocks, kThreads, 0, stream>>>(xp, w.data_ptr<scalar_t>(), op, m,
                                                              static_cast<int>(k));
    else
      conv1d_const<scalar_t><<<blocks, kThreads, 0, stream>>>(xp, op, m, static_cast<int>(k));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
