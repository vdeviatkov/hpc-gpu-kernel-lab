#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "lab/launch.cuh"
#include "lab/vector.cuh"

// Variants form a 2x2 design: {scalar, 16-byte vector} x {one item per thread, capped
// grid-stride loop}. Comparing along one axis changes exactly one factor.
enum Variant { kScalar = 0, kGridStride = 1, kVecGridStride = 2, kVecDirect = 3 };
constexpr int64_t kMaxGridStrideBlocks = 4096;

// The bindings reject overlap between the output and either input, so out may be marked
// __restrict__. Inputs may alias each other; both are read-only, which restrict permits.
template <typename T> __device__ __forceinline__ T add_fp32(T x, T y) {
  return static_cast<T>(static_cast<float>(x) + static_cast<float>(y));
}

using lab::Pack;

// One 16-byte pack: four FP32 or eight FP16/BF16 elements, one 128-bit load per input and
// one 128-bit store (see lab/vector.cuh). The launcher checks alignment first.
template <typename T>
__device__ __forceinline__ void add_pack(const T *__restrict__ a, const T *__restrict__ b,
                                         T *__restrict__ out, int64_t i) {
  const Pack<T> x = lab::load_pack(a, i);
  const Pack<T> y = lab::load_pack(b, i);
  Pack<T> z;
#pragma unroll
  for (int k = 0; k < Pack<T>::kSize; ++k)
    z.v[k] = add_fp32(x.v[k], y.v[k]);
  lab::store_pack(out, i, z);
}

template <typename T>
__global__ void scalar_coalesced(const T *__restrict__ a, const T *__restrict__ b,
                                 T *__restrict__ out, int64_t n) {
  const int64_t i = lab::global_index();
  if (i < n)
    out[i] = add_fp32(a[i], b[i]);
}

template <typename T>
__global__ void grid_stride(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out,
                            int64_t n) {
  const int64_t stride = lab::grid_threads();
  for (int64_t i = lab::global_index(); i < n; i += stride)
    out[i] = add_fp32(a[i], b[i]);
}

// Vector bulk in a capped grid-stride loop; the scalar tail (fewer than one pack) is
// handled once by the first threads, in the same launch.
template <typename T>
__global__ void vec_grid_stride(const T *__restrict__ a, const T *__restrict__ b,
                                T *__restrict__ out, int64_t n) {
  const int64_t tid = lab::global_index();
  const int64_t stride = lab::grid_threads();
  const int64_t packs = n / Pack<T>::kSize;
  for (int64_t i = tid; i < packs; i += stride)
    add_pack(a, b, out, i);
  const int64_t tail = packs * Pack<T>::kSize + tid;
  if (tail < n)
    out[tail] = add_fp32(a[tail], b[tail]);
}

// One pack per thread, no loop: the vector counterpart of scalar_coalesced.
template <typename T>
__global__ void vec_direct(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out,
                           int64_t n) {
  const int64_t i = lab::global_index();
  const int64_t packs = n / Pack<T>::kSize;
  if (i < packs) {
    add_pack(a, b, out, i);
  } else if (i == packs) {
    // Fewer than one pack, once per launch; unrolling would only inflate registers.
#pragma unroll 1
    for (int64_t j = packs * Pack<T>::kSize; j < n; ++j)
      out[j] = add_fp32(a[j], b[j]);
  }
}

void launch_add(const at::Tensor &a, const at::Tensor &b, at::Tensor &out, int variant,
                int threads) {
  const c10::cuda::CUDAGuard guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const int64_t n = a.numel();
  // Misaligned buffers fall back along the vector axis only: vec_grid_stride -> grid_stride,
  // vec_direct -> scalar_coalesced. The loop structure is preserved.
  const bool aligned = lab::pack_aligned({a.data_ptr(), b.data_ptr(), out.data_ptr()});
  if ((variant == kVecGridStride || variant == kVecDirect) && !aligned)
    variant = variant == kVecGridStride ? kGridStride : kScalar;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, a.scalar_type(), "vector_add", [&] {
    const auto *pa = a.data_ptr<scalar_t>();
    const auto *pb = b.data_ptr<scalar_t>();
    auto *po = out.data_ptr<scalar_t>();
    const int64_t packs = n / Pack<scalar_t>::kSize;
    switch (variant) {
    case kScalar:
      scalar_coalesced<<<lab::grid_blocks(n, threads), threads, 0, stream>>>(pa, pb, po, n);
      break;
    case kGridStride:
      grid_stride<<<lab::grid_blocks(n, threads, kMaxGridStrideBlocks), threads, 0, stream>>>(
          pa, pb, po, n);
      break;
    case kVecGridStride:
      // At least one block so the tail runs when n is smaller than one pack.
      vec_grid_stride<<<lab::grid_blocks(std::max<int64_t>(packs, 1), threads,
                                         kMaxGridStrideBlocks),
                        threads, 0, stream>>>(pa, pb, po, n);
      break;
    default:
      vec_direct<<<lab::grid_blocks(packs + 1, threads), threads, 0, stream>>>(pa, pb, po, n);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
