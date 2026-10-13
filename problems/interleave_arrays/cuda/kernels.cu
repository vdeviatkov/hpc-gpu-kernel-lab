#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lab/launch.cuh"
#include "lab/vector.cuh"

using lab::Pack;

// Variants differ in how threads map to the data: by input element, by output element,
// by input element with one paired store, or by 16-byte packs of both.
enum Variant { kInput = 0, kOutput = 1, kPair = 2, kVec = 3 };

// Thread i reads a[i] and b[i] (contiguous across the warp) and writes out[2i] and
// out[2i + 1] (stride-2 stores).
template <typename T>
__global__ void interleave_input(const T *__restrict__ a, const T *__restrict__ b,
                                 T *__restrict__ out, int64_t n) {
  const int64_t i = lab::global_index();
  if (i >= n)
    return;
  out[i * 2] = a[i];
  out[i * 2 + 1] = b[i];
}

// Thread i writes out[i] (contiguous stores) and reads from a or b depending on whether i
// is even or odd, so neighbouring lanes read alternately from the two inputs.
template <typename T>
__global__ void interleave_output(const T *__restrict__ a, const T *__restrict__ b,
                                  T *__restrict__ out, int64_t n) {
  const int64_t i = lab::global_index();
  if (i >= 2 * n)
    return;
  out[i] = i & 1 ? b[i >> 1] : a[i >> 1];
}

// Thread i writes {a[i], b[i]} as one aligned 2-element store (8 bytes for FP32, 4 for
// FP16/BF16), so every store instruction is contiguous across the warp. The output is
// freshly allocated, hence always aligned for pairs.
template <typename T>
__global__ void interleave_pair(const T *__restrict__ a, const T *__restrict__ b,
                                T *__restrict__ out, int64_t n) {
  const int64_t i = lab::global_index();
  if (i >= n)
    return;
  struct alignas(2 * sizeof(T)) Pair {
    T a;
    T b;
  };
  reinterpret_cast<Pair *>(out)[i] = Pair{a[i], b[i]};
}

// Thread i reads pack i of a and of b (k elements each) and writes output packs 2i and
// 2i + 1, which hold a0 b0 a1 b1 ... in order: element j goes to output pack j / (k / 2),
// slot 2 * (j % (k / 2)). The thread just past the last full pack handles the remaining
// n % k elements. The launcher checks that a, b and out are 16-byte aligned.
template <typename T>
__global__ void interleave_vec(const T *__restrict__ a, const T *__restrict__ b,
                               T *__restrict__ out, int64_t n) {
  constexpr int k = Pack<T>::kSize;
  constexpr int half = k / 2;
  const int64_t i = lab::global_index();
  const int64_t packs = n / k;
  if (i < packs) {
    const Pack<T> va = lab::load_pack(a, i);
    const Pack<T> vb = lab::load_pack(b, i);
    Pack<T> results[2];
#pragma unroll
    for (int j = 0; j < k; ++j) {
      results[j / half].v[2 * (j % half)] = va.v[j];
      results[j / half].v[2 * (j % half) + 1] = vb.v[j];
    }
    lab::store_pack(out, 2 * i, results[0]);
    lab::store_pack(out, 2 * i + 1, results[1]);
  } else if (i == packs) {
#pragma unroll 1
    for (int64_t j = packs * k; j < n; ++j) {
      out[2 * j] = a[j];
      out[2 * j + 1] = b[j];
    }
  }
}

void launch_interleave(const at::Tensor &a, const at::Tensor &b, at::Tensor out, int64_t variant,
                       int64_t threads) {
  const c10::cuda::CUDAGuard guard(a.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const int64_t n = a.numel();
  if (n == 0)
    return;
  const int t = static_cast<int>(threads);
  // Misaligned inputs (offset slices) fall back to the input-mapped kernel.
  if (variant == kVec && !lab::pack_aligned({a.data_ptr(), b.data_ptr(), out.data_ptr()}))
    variant = kInput;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, a.scalar_type(), "interleave", [&] {
    const auto *pa = a.data_ptr<scalar_t>();
    const auto *pb = b.data_ptr<scalar_t>();
    auto *po = out.data_ptr<scalar_t>();
    switch (variant) {
    case kInput:
      interleave_input<<<lab::grid_blocks(n, t), t, 0, stream>>>(pa, pb, po, n);
      break;
    case kOutput:
      interleave_output<<<lab::grid_blocks(2 * n, t), t, 0, stream>>>(pa, pb, po, n);
      break;
    case kPair:
      interleave_pair<<<lab::grid_blocks(n, t), t, 0, stream>>>(pa, pb, po, n);
      break;
    default:
      interleave_vec<<<lab::grid_blocks(n / Pack<scalar_t>::kSize + 1, t), t, 0, stream>>>(pa, pb,
                                                                                           po, n);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
