#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

// Variants differ only in how the condition is written (select, branch, fmax) or in the
// load width (vec). All use one element (or one 16-byte pack) per thread.
enum Variant { kSelect = 0, kBranch = 1, kFmax = 2, kVec = 3 };

template <typename T> struct alignas(16) Pack {
  static constexpr int kSize = 16 / sizeof(T);
  T v[kSize];
};

// Contract formula: only strictly negative values become 0, so -0.0 and NaN pass through
// unchanged, exactly like torch.relu.
template <typename T> __device__ __forceinline__ T relu_value(T v) {
  return static_cast<float>(v) < 0.0f ? T(0) : v;
}

template <typename T>
__global__ void relu_select(const T *__restrict__ x, T *__restrict__ out, int64_t n) {
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  out[idx] = relu_value(x[idx]);
}

template <typename T>
__global__ void relu_branch(const T *__restrict__ x, T *__restrict__ out, int64_t n) {
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
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
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  out[idx] = T(fmaxf(static_cast<float>(x[idx]), 0.0f));
}

// One 16-byte pack per thread; the thread just past the last full pack handles the
// remaining n % kSize elements. Reading T arrays through Pack<T> is the usual CUDA
// vector-access idiom (see problems/vector_add/cuda/kernels.cu for the aliasing note).
template <typename T>
__global__ void relu_vec(const T *__restrict__ x, T *__restrict__ out, int64_t n) {
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t packs = n / Pack<T>::kSize;
  if (idx < packs) {
    const Pack<T> val = reinterpret_cast<const Pack<T> *>(x)[idx];
    Pack<T> result;
#pragma unroll
    for (int i = 0; i < Pack<T>::kSize; ++i)
      result.v[i] = relu_value(val.v[i]);
    reinterpret_cast<Pack<T> *>(out)[idx] = result;
  } else if (idx == packs) {
#pragma unroll 1
    for (int64_t j = packs * Pack<T>::kSize; j < n; ++j)
      out[j] = relu_value(x[j]);
  }
}

static int grid_blocks(int64_t items, int threads) {
  const int64_t blocks = (items + threads - 1) / threads;
  TORCH_CHECK(blocks <= 2147483647LL, "relu: input exceeds the launch grid limit");
  return static_cast<int>(blocks);
}

void launch_relu(const at::Tensor &x, at::Tensor out, int64_t variant, int64_t threads) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t n = x.numel();
  if (n == 0)
    return;
  const auto addresses =
      reinterpret_cast<uintptr_t>(x.data_ptr()) | reinterpret_cast<uintptr_t>(out.data_ptr());
  // Misaligned buffers fall back to the scalar select, the same formula.
  if (variant == kVec && addresses % 16 != 0)
    variant = kSelect;
  const int t = static_cast<int>(threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "relu", [&] {
    const auto *px = x.data_ptr<scalar_t>();
    auto *po = out.data_ptr<scalar_t>();
    switch (variant) {
    case kSelect:
      relu_select<<<grid_blocks(n, t), t, 0, stream>>>(px, po, n);
      break;
    case kBranch:
      relu_branch<<<grid_blocks(n, t), t, 0, stream>>>(px, po, n);
      break;
    case kFmax:
      relu_fmax<<<grid_blocks(n, t), t, 0, stream>>>(px, po, n);
      break;
    default:
      relu_vec<<<grid_blocks(n / Pack<scalar_t>::kSize + 1, t), t, 0, stream>>>(px, po, n);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
