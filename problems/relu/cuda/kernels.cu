#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lab/launch.cuh"
#include "lab/vector.cuh"

// Variants differ only in how the condition is written (select, branch, fmax) or in the
// load width (vec). All use one element (or one 16-byte pack) per thread.
enum Variant { kSelect = 0, kBranch = 1, kFmax = 2, kVec = 3 };

using lab::Pack;

// Contract formula: only strictly negative values become 0, so -0.0 and NaN pass through
// unchanged, exactly like torch.relu.
template <typename T> __device__ __forceinline__ T relu_value(T v) {
  return static_cast<float>(v) < 0.0f ? T(0) : v;
}

template <typename T>
__global__ void relu_select(const T *__restrict__ x, T *__restrict__ out, int64_t n) {
  const int64_t idx = lab::global_index();
  if (idx >= n)
    return;

  out[idx] = relu_value(x[idx]);
}

template <typename T>
__global__ void relu_branch(const T *__restrict__ x, T *__restrict__ out, int64_t n) {
  const int64_t idx = lab::global_index();
  if (idx >= n)
    return;

  const T v = x[idx];
  if (static_cast<float>(v) < 0.0f) {
    out[idx] = T(0);
  } else {
    out[idx] = v;
  }
}

// fmaxf returns the non-NaN operand and +0.0 for (-0.0, 0.0), so this variant differs
// from torch.relu on NaN and -0.0. Kept to compare its instructions with the select.
template <typename T>
__global__ void relu_fmax(const T *__restrict__ x, T *__restrict__ out, int64_t n) {
  const int64_t idx = lab::global_index();
  if (idx >= n)
    return;

  out[idx] = T(fmaxf(static_cast<float>(x[idx]), 0.0f));
}

// One 16-byte pack per thread (lab/vector.cuh); the thread just past the last full pack
// handles the remaining n % kSize elements.
template <typename T>
__global__ void relu_vec(const T *__restrict__ x, T *__restrict__ out, int64_t n) {
  const int64_t idx = lab::global_index();
  const int64_t packs = n / Pack<T>::kSize;
  if (idx < packs) {
    const Pack<T> val = lab::load_pack(x, idx);
    Pack<T> result;
#pragma unroll
    for (int i = 0; i < Pack<T>::kSize; ++i)
      result.v[i] = relu_value(val.v[i]);
    lab::store_pack(out, idx, result);
  } else if (idx == packs) {
#pragma unroll 1
    for (int64_t j = packs * Pack<T>::kSize; j < n; ++j)
      out[j] = relu_value(x[j]);
  }
}

void launch_relu(const at::Tensor &x, at::Tensor out, int64_t variant, int64_t threads) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t n = x.numel();
  if (n == 0)
    return;
  // Misaligned buffers fall back to the scalar select, the same formula.
  if (variant == kVec && !lab::pack_aligned({x.data_ptr(), out.data_ptr()}))
    variant = kSelect;
  const int t = static_cast<int>(threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "relu", [&] {
    const auto *px = x.data_ptr<scalar_t>();
    auto *po = out.data_ptr<scalar_t>();
    switch (variant) {
    case kSelect:
      relu_select<<<lab::grid_blocks(n, t), t, 0, stream>>>(px, po, n);
      break;
    case kBranch:
      relu_branch<<<lab::grid_blocks(n, t), t, 0, stream>>>(px, po, n);
      break;
    case kFmax:
      relu_fmax<<<lab::grid_blocks(n, t), t, 0, stream>>>(px, po, n);
      break;
    default:
      relu_vec<<<lab::grid_blocks(n / Pack<scalar_t>::kSize + 1, t), t, 0, stream>>>(px, po, n);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
