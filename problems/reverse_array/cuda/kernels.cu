#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda/std/utility>
#include <cuda_runtime.h>

#include <cstdint>

#include "lab/launch.cuh"
#include "lab/vector.cuh"

using lab::Pack;

enum Variant { kPair = 0, kVec = 1 };

// One thread owns one pair (i, n - 1 - i): it reads both elements before writing either,
// so no other thread can observe a half-swapped pair. For odd n the middle element has no
// owner and stays in place.
template <typename T> __global__ void reverse_pair(T *__restrict__ x, int64_t n) {
  const int64_t i = lab::global_index();
  if (i >= n / 2)
    return;
  cuda::std::swap(x[i], x[n - 1 - i]);
}

// One thread owns one pair of 16-byte packs: front pack i and its mirror, pack
// packs - 1 - i. The mirror starts on a pack boundary only when n is a multiple of
// Pack<T>::kSize, so the launcher requires that (and a 16-byte aligned pointer) and
// otherwise uses reverse_pair. Swapping element j of the front pack with element k - 1 - j
// of the back pack, for every j, reverses both packs and exchanges them. With an odd number
// of packs, the middle pack mirrors onto itself and is reversed in place by one extra thread.
template <typename T> __global__ void reverse_vec(T *__restrict__ x, int64_t n) {
  constexpr int k = Pack<T>::kSize;
  const int64_t i = lab::global_index();
  const int64_t packs = n / k;
  const int64_t pairs = packs / 2;
  if (i < pairs) {
    const int64_t j = packs - 1 - i;
    Pack<T> front = lab::load_pack(x, i);
    Pack<T> back = lab::load_pack(x, j);
#pragma unroll
    for (int e = 0; e < k; ++e)
      cuda::std::swap(front.v[e], back.v[k - 1 - e]);
    lab::store_pack(x, i, front);
    lab::store_pack(x, j, back);
  } else if (i == pairs && packs % 2 == 1) {
    Pack<T> middle = lab::load_pack(x, pairs);
#pragma unroll
    for (int e = 0; e < k / 2; ++e)
      cuda::std::swap(middle.v[e], middle.v[k - 1 - e]);
    lab::store_pack(x, pairs, middle);
  }
}

void launch_reverse_(at::Tensor x, int64_t variant, int64_t threads) {
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t n = x.numel();
  if (n < 2)
    return; // Nothing to swap; a zero-block launch would be an error.
  const int t = static_cast<int>(threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "reverse_", [&] {
    auto *px = x.data_ptr<scalar_t>();
    constexpr int k = Pack<scalar_t>::kSize;
    const bool vec_ok = n % k == 0 && lab::pack_aligned({x.data_ptr()});
    if (variant == kVec && vec_ok) {
      const int64_t pairs = (n / k) / 2;
      reverse_vec<<<lab::grid_blocks(pairs + 1, t), t, 0, stream>>>(px, n);
    } else {
      reverse_pair<<<lab::grid_blocks(n / 2, t), t, 0, stream>>>(px, n);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
