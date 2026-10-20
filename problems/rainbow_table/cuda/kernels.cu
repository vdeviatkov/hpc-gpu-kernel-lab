// Variants: int64 reference, uint32 baseline, 2/4 interleaved chains,
// aligned 16-byte loads/stores, and specialized unrolled round counts.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lab/launch.cuh"
#include "lab/vector.cuh"

template <typename T> __device__ __forceinline__ T fnv1a(T value) {
  T h = T(2166136261u);
#pragma unroll
  for (int byte = 0; byte < 4; ++byte) {
    h = ((h ^ ((value >> (8 * byte)) & T(255))) * T(16777619)) & T(0xffffffffu);
  }
  return h;
}

template <typename T>
__global__ void scalar_kernel(const int32_t *__restrict__ x, int64_t rounds,
                              uint32_t *__restrict__ out, int64_t n) {
  const int64_t i = lab::global_index();
  if (i < n) {
    T value = static_cast<uint32_t>(x[i]);
#pragma unroll 1
    for (int64_t r = 0; r < rounds; ++r)
      value = fnv1a(value);
    out[i] = static_cast<uint32_t>(value);
  }
}

// Strided lanes keep each warp's memory accesses contiguous. The inner loop
// exposes independent chains to the instruction scheduler.
template <int K, bool Vector>
__global__ void ilp_kernel(const int32_t *__restrict__ x, int64_t rounds,
                           uint32_t *__restrict__ out, int64_t n) {
  const int64_t thread = lab::global_index();
  const int64_t base = Vector ? thread * K : int64_t(blockIdx.x) * blockDim.x * K + threadIdx.x;
  uint32_t values[K] = {};
  if constexpr (Vector) {
    if (base + K <= n) {
      const auto pack = lab::load_pack(x, thread);
#pragma unroll
      for (int k = 0; k < K; ++k)
        values[k] = static_cast<uint32_t>(pack.v[k]);
    } else {
#pragma unroll
      for (int k = 0; k < K; ++k)
        if (base + k < n)
          values[k] = static_cast<uint32_t>(x[base + k]);
    }
  } else {
#pragma unroll
    for (int k = 0; k < K; ++k)
      if (base + int64_t(k) * blockDim.x < n)
        values[k] = static_cast<uint32_t>(x[base + int64_t(k) * blockDim.x]);
  }
#pragma unroll 1
  for (int64_t r = 0; r < rounds; ++r) {
    uint32_t h[K];
#pragma unroll
    for (int k = 0; k < K; ++k)
      h[k] = 2166136261u;
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
#pragma unroll
      for (int k = 0; k < K; ++k)
        h[k] = (h[k] ^ ((values[k] >> (8 * byte)) & 255u)) * 16777619u;
    }
#pragma unroll
    for (int k = 0; k < K; ++k)
      values[k] = h[k];
  }
  if constexpr (Vector) {
    if (base + K <= n) {
      lab::Pack<uint32_t> pack;
#pragma unroll
      for (int k = 0; k < K; ++k)
        pack.v[k] = values[k];
      lab::store_pack(out, thread, pack);
    } else {
#pragma unroll
      for (int k = 0; k < K; ++k)
        if (base + k < n)
          out[base + k] = values[k];
    }
  } else {
#pragma unroll
    for (int k = 0; k < K; ++k)
      if (base + int64_t(k) * blockDim.x < n)
        out[base + int64_t(k) * blockDim.x] = values[k];
  }
}

template <int R>
__global__ void unrolled_kernel(const int32_t *__restrict__ x, uint32_t *__restrict__ out,
                                int64_t n) {
  const int64_t i = lab::global_index();
  if (i < n) {
    uint32_t value = static_cast<uint32_t>(x[i]);
#pragma unroll
    for (int r = 0; r < R; ++r)
      value = fnv1a(value);
    out[i] = value;
  }
}

void launch_rainbow_table(const at::Tensor &x, int64_t rounds, at::Tensor out, int64_t variant,
                          int64_t threads) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t n = x.numel();
  if (!n)
    return;
  const auto *input = x.data_ptr<int32_t>();
  auto *output = out.data_ptr<uint32_t>();
  if (variant == 4 && !lab::pack_aligned({input, output}))
    variant = 3;
  const int items = variant == 2 ? 2 : (variant == 3 || variant == 4 ? 4 : 1);
  const int blocks = lab::grid_blocks((n + items - 1) / items, threads);
  switch (variant) {
  case 0:
    scalar_kernel<int64_t><<<blocks, threads, 0, stream>>>(input, rounds, output, n);
    break;
  case 2:
    ilp_kernel<2, false><<<blocks, threads, 0, stream>>>(input, rounds, output, n);
    break;
  case 3:
    ilp_kernel<4, false><<<blocks, threads, 0, stream>>>(input, rounds, output, n);
    break;
  case 4:
    ilp_kernel<4, true><<<blocks, threads, 0, stream>>>(input, rounds, output, n);
    break;
  case 5:
#define LAUNCH_ROUNDS(R)                                                                           \
  case R:                                                                                          \
    unrolled_kernel<R><<<blocks, threads, 0, stream>>>(input, output, n);                          \
    break
    switch (rounds) {
      LAUNCH_ROUNDS(0);
      LAUNCH_ROUNDS(1);
      LAUNCH_ROUNDS(2);
      LAUNCH_ROUNDS(3);
      LAUNCH_ROUNDS(5);
      LAUNCH_ROUNDS(10);
      LAUNCH_ROUNDS(20);
      LAUNCH_ROUNDS(50);
      LAUNCH_ROUNDS(100);
    default:
      scalar_kernel<uint32_t><<<blocks, threads, 0, stream>>>(input, rounds, output, n);
    }
#undef LAUNCH_ROUNDS
    break;
  default:
    scalar_kernel<uint32_t><<<blocks, threads, 0, stream>>>(input, rounds, output, n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
