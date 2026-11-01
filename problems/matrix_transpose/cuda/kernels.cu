// 08 · Matrix Transpose: CUDA kernels.
//
// A transpose never interprets its data, so the kernels move unsigned integers of the
// element's width: one instantiation serves FP32 and one serves FP16 and BF16, and bit
// patterns (NaN payloads, -0.0, subnormals) pass through unchanged by construction.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lab/launch.cuh"
#include "lab/vector.cuh"

namespace {

// One block transposes a tile of kTileM input rows by kTileN input columns. The shared
// tile holds it already transposed (kTileN x kTileM), so each output row segment is
// kTileM consecutive shared elements.
constexpr int kTileM = 16;
constexpr int kTileN = 256;
constexpr int kThreads = 256;
// gridDim.y is limited to 65535.
constexpr int64_t kMaxGridY = 65535;

// Flat shared index of tile[row][col]. A 16-element row is half of a 32-bank line, so
// the XOR acts on the bank-word index within a 128-byte line (rows 2 FP32 / 4 FP16 apart
// share one). The key, from row / E, spreads the load phase's lanes (rows 4 apart) and the
// store phase's (adjacent rows) over all 32 banks. Modelled (32 banks x 4 B): both phases
// conflict-free for FP32 and FP16, against 32-way without it.
template <typename T> __device__ __forceinline__ int swizzle(int row, int col) {
  constexpr int E = 4 / static_cast<int>(sizeof(T)); // elements per 4-byte bank word
  const int w = row / E;
  return (row * kTileM + col) ^ ((((w >> 1) ^ (w >> 2)) & 31) * E);
}

// Interior tiles move 16-byte packs: each thread issues kTileM * kTileN / K / kThreads
// independent packs per phase (4 FP32, 2 FP16). The trip count is a compile-time
// constant, so the loops unroll and the loads are all in flight together. Edge tiles, and
// inputs whose rows are not pack-aligned (`vec` false), fall back to bounds-checked scalars.
// `full` is uniform per block, so the barrier is reached by every thread.
//
// Shared memory is swizzled (see swizzle()). Unswizzled, the load phase's neighbouring
// lanes write tile[c + k][r] with c four apart, 64 words apart: the same bank, a 32-way
// conflict. The swizzle splits each pack's 4 contiguous shared words, so the store phase
// reads them with scalar accesses instead of a possible single 128-bit one.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    transpose_vec_ilp(const T *__restrict__ a, T *__restrict__ out, int64_t m, int64_t n,
                      bool vec) {
  constexpr int K = lab::Pack<T>::kSize;
  constexpr int kPacksPerThread = kTileM * kTileN / K / kThreads;
  static_assert(kTileM % K == 0 && kTileN % K == 0 &&
                    kTileM * kTileN / K == kPacksPerThread * kThreads,
                "tile must split into whole packs per thread");
  __shared__ T tile[kTileN * kTileM];

  const int64_t row0 = int64_t(blockIdx.y) * kTileM;
  const int64_t col0 = int64_t(blockIdx.x) * kTileN;
  const bool full = vec && row0 + kTileM <= m && col0 + kTileN <= n;

  if (full) {
    // Lanes walk along an input row: a warp reads 32 consecutive packs.
#pragma unroll
    for (int j = 0; j < kPacksPerThread; ++j) {
      const int i = threadIdx.x + j * kThreads;
      const int r = i / (kTileN / K);
      const int c = i % (kTileN / K) * K;
      const lab::Pack<T> p = lab::load_pack(a, ((row0 + r) * n + col0 + c) / K);
#pragma unroll
      for (int k = 0; k < K; ++k)
        tile[swizzle<T>(c + k, r)] = p.v[k];
    }
  } else {
    for (int i = threadIdx.x; i < kTileM * kTileN; i += kThreads) {
      const int r = i / kTileN;
      const int c = i % kTileN;
      if (row0 + r < m && col0 + c < n)
        tile[swizzle<T>(c, r)] = a[(row0 + r) * n + col0 + c];
    }
  }
  __syncthreads();

  // Output is n x m: tile row c is output row col0 + c, tile column r is output column row0 + r.
  if (full) {
#pragma unroll
    for (int j = 0; j < kPacksPerThread; ++j) {
      const int i = threadIdx.x + j * kThreads;
      const int c = i / (kTileM / K);
      const int r = i % (kTileM / K) * K;
      lab::Pack<T> p;
#pragma unroll
      for (int k = 0; k < K; ++k)
        p.v[k] = tile[swizzle<T>(c, r + k)];
      lab::store_pack(out, ((col0 + c) * m + row0 + r) / K, p);
    }
  } else {
    for (int i = threadIdx.x; i < kTileN * kTileM; i += kThreads) {
      const int c = i / kTileM;
      const int r = i % kTileM;
      if (col0 + c < n && row0 + r < m)
        out[(col0 + c) * m + row0 + r] = tile[swizzle<T>(c, r)];
    }
  }
}

template <typename T>
void launch(const T *a, T *out, int64_t m, int64_t n, cudaStream_t stream) {
  constexpr int64_t K = lab::Pack<T>::kSize;
  // Every input and output row must start on a 16-byte boundary for pack access.
  const bool vec = lab::pack_aligned({a, out}) && n % K == 0 && m % K == 0;
  const int64_t tiles_m = (m + kTileM - 1) / kTileM;
  TORCH_CHECK(tiles_m <= kMaxGridY, "transpose: too many rows for the launch grid");
  const dim3 grid(lab::grid_blocks(n, kTileN), static_cast<unsigned>(tiles_m));
  transpose_vec_ilp<<<grid, kThreads, 0, stream>>>(a, out, m, n, vec);
}

} // namespace

void launch_transpose(const at::Tensor &a, at::Tensor out) {
  const c10::cuda::CUDAGuard guard(a.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const int64_t m = a.size(0);
  const int64_t n = a.size(1);
  if (!a.numel())
    return;
  if (a.element_size() == 4)
    launch(static_cast<const uint32_t *>(a.data_ptr()), static_cast<uint32_t *>(out.data_ptr()), m,
           n, stream);
  else
    launch(static_cast<const uint16_t *>(a.data_ptr()), static_cast<uint16_t *>(out.data_ptr()), m,
           n, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
