#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda/std/utility>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lab/launch.cuh"
#include "lab/vector.cuh"

using lab::Pack;

enum Variant { kPair = 0, kVec = 1, kTile = 2, kTileAsync = 3 };

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

// ---- Tiled reversal through shared memory, for any n ---------------------------------
//
// Block b owns front range F = [b*T, b*T + len) and its mirror B = [n - b*T - len,
// n - b*T), with T = blockDim.x * k elements. The ranges of all blocks partition the pairs,
// so every element belongs to exactly one block. Each range is moved with 16-byte accesses
// for the packs that lie entirely inside it and scalar accesses for the at most k - 1
// elements at either end; a pack that straddles two blocks' ranges is only ever touched
// element by element, by its owners. The back range's alignment depends on n, but only its
// ends are affected, so wide accesses work for any n (given an aligned base pointer).

// Copies x[start, start + len) into buf[0, len).
template <typename T>
__device__ __forceinline__ void tile_load(const T *x, int64_t start, int64_t len, T *buf) {
  constexpr int k = Pack<T>::kSize;
  const int64_t first = (start + k - 1) / k * k; // first pack boundary inside the range
  const int64_t last = (start + len) / k * k;    // end of the last full pack
  if (first >= last) {                           // no full pack: scalar only
    for (int64_t t = threadIdx.x; t < len; t += blockDim.x)
      buf[t] = x[start + t];
    return;
  }
  const int64_t head = first - start; // < k elements before the first pack
  const int64_t tail = last - start;  // offset of the < k elements after the last pack
  if (threadIdx.x < head)
    buf[threadIdx.x] = x[start + threadIdx.x];
  if (threadIdx.x < len - tail)
    buf[tail + threadIdx.x] = x[last + threadIdx.x];
  for (int64_t p = threadIdx.x; p < (last - first) / k; p += blockDim.x) {
    const Pack<T> v = lab::load_pack(x, first / k + p);
#pragma unroll
    for (int e = 0; e < k; ++e)
      buf[head + p * k + e] = v.v[e];
  }
}

// Writes x[start + t] = src[len - 1 - t] for t in [0, len): the source range, reversed.
template <typename T>
__device__ __forceinline__ void tile_store_reversed(T *x, int64_t start, int64_t len,
                                                    const T *src) {
  constexpr int k = Pack<T>::kSize;
  const int64_t first = (start + k - 1) / k * k;
  const int64_t last = (start + len) / k * k;
  if (first >= last) {
    for (int64_t t = threadIdx.x; t < len; t += blockDim.x)
      x[start + t] = src[len - 1 - t];
    return;
  }
  const int64_t head = first - start;
  const int64_t tail = last - start;
  if (threadIdx.x < head)
    x[start + threadIdx.x] = src[len - 1 - threadIdx.x];
  if (threadIdx.x < len - tail)
    x[last + threadIdx.x] = src[len - 1 - (tail + threadIdx.x)];
  for (int64_t p = threadIdx.x; p < (last - first) / k; p += blockDim.x) {
    Pack<T> v;
#pragma unroll
    for (int e = 0; e < k; ++e)
      v.v[e] = src[len - 1 - (head + p * k + e)];
    lab::store_pack(x, first / k + p, v);
  }
}

// Dynamic shared memory: two buffers of blockDim.x * k elements (2 * blockDim.x * 16 bytes).
template <typename T> __global__ void reverse_tile(T *__restrict__ x, int64_t n) {
  constexpr int k = Pack<T>::kSize;
  extern __shared__ __align__(16) unsigned char shared[];
  const int64_t tile = int64_t(blockDim.x) * k;
  T *front_buf = reinterpret_cast<T *>(shared);
  T *back_buf = front_buf + tile;
  const int64_t front = int64_t(blockIdx.x) * tile;
  const int64_t remaining = n / 2 - front;
  const int64_t len = remaining < tile ? remaining : tile;
  const int64_t back = n - front - len;
  tile_load(x, front, len, front_buf);
  tile_load(x, back, len, back_buf);
  __syncthreads(); // the whole block has read both ranges before anyone writes
  tile_store_reversed(x, front, len, back_buf);
  tile_store_reversed(x, back, len, front_buf);
}

// ---- Same tiling, with asynchronous global -> shared copies (cp.async, SM80+) -----------
//
// Interior 16-byte packs are copied by __pipeline_memcpy_async straight into shared memory,
// without passing through registers; the < k elements at either end use ordinary loads
// (cp.async cannot copy 2-byte elements). cp.async needs a 16-byte aligned shared-memory
// destination, so element t of a range is stored at buf[start % k + t]: interior packs
// then sit on pack boundaries in shared memory as they do in global memory. Each buffer
// therefore holds tile + k elements.

// Starts the copy of x[start, start + len) into buf; returns the offset where element 0 of
// the range lands. The caller waits with __pipeline_wait_prior(0) and __syncthreads().
template <typename T>
__device__ __forceinline__ int64_t tile_load_async(const T *x, int64_t start, int64_t len, T *buf) {
  constexpr int k = Pack<T>::kSize;
  const int64_t offset = start % k;
  const int64_t first = (start + k - 1) / k * k;
  const int64_t last = (start + len) / k * k;
  if (first >= last) {
    for (int64_t t = threadIdx.x; t < len; t += blockDim.x)
      buf[offset + t] = x[start + t];
    return offset;
  }
  const int64_t head = first - start;
  const int64_t tail = last - start;
  if (threadIdx.x < head)
    buf[offset + threadIdx.x] = x[start + threadIdx.x];
  if (threadIdx.x < len - tail)
    buf[offset + tail + threadIdx.x] = x[last + threadIdx.x];
  for (int64_t p = threadIdx.x; p < (last - first) / k; p += blockDim.x)
    __pipeline_memcpy_async(buf + offset + head + p * k, x + first + p * k, lab::kVectorBytes);
  __pipeline_commit();
  return offset;
}

// Dynamic shared memory: two buffers of blockDim.x * k + k elements
// (2 * (blockDim.x + 1) * 16 bytes).
template <typename T> __global__ void reverse_tile_async(T *__restrict__ x, int64_t n) {
  constexpr int k = Pack<T>::kSize;
  extern __shared__ __align__(16) unsigned char shared[];
  const int64_t tile = int64_t(blockDim.x) * k;
  T *front_buf = reinterpret_cast<T *>(shared);
  T *back_buf = front_buf + tile + k;
  const int64_t front = int64_t(blockIdx.x) * tile;
  const int64_t remaining = n / 2 - front;
  const int64_t len = remaining < tile ? remaining : tile;
  const int64_t back = n - front - len;
  const int64_t front_offset = tile_load_async(x, front, len, front_buf);
  const int64_t back_offset = tile_load_async(x, back, len, back_buf);
  __pipeline_wait_prior(0); // this thread's async copies have landed
  __syncthreads();          // and so have everyone else's, plus the ordinary loads
  tile_store_reversed(x, front, len, back_buf + back_offset);
  tile_store_reversed(x, back, len, front_buf + front_offset);
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
    const bool aligned = lab::pack_aligned({x.data_ptr()});
    if (variant == kVec && aligned && n % k == 0) {
      const int64_t pairs = (n / k) / 2;
      reverse_vec<<<lab::grid_blocks(pairs + 1, t), t, 0, stream>>>(px, n);
    } else if (variant == kTile && aligned) {
      const size_t shared_bytes = 2 * size_t(t) * lab::kVectorBytes;
      reverse_tile<<<lab::grid_blocks(n / 2, t * k), t, shared_bytes, stream>>>(px, n);
    } else if (variant == kTileAsync && aligned) {
      const size_t shared_bytes = 2 * (size_t(t) + 1) * lab::kVectorBytes;
      reverse_tile_async<<<lab::grid_blocks(n / 2, t * k), t, shared_bytes, stream>>>(px, n);
    } else {
      reverse_pair<<<lab::grid_blocks(n / 2, t), t, 0, stream>>>(px, n);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
