// 09 · 1D Convolution: CUDA kernels.
//
// out[i] = sum_j x[i + j] * w[j] for i < m = N - K + 1, accumulated in FP32 and rounded once
// to the input dtype. One thread per output in every variant:
//
//   0 conv1d_naive  x and w read from global memory. Per tap a warp issues one coalesced
//                   (but j-misaligned) x load and one broadcast w load; L1 absorbs the
//                   K-fold reuse of x, so the LSU/L1 pipe, not DRAM, is the expected limit.
//   1 conv1d_const  w converted to FP32 and staged in __constant__ memory. Every lane reads
//                   the same tap, which the constant cache broadcasts, so the w loads leave
//                   the LSU/L1 pipe; x loads are unchanged. Capacity: kMaxConstTaps taps.
//   2 conv1d_smem   the block stages kThreads + taps - 1 x values in shared memory behind a
//                   barrier, then each thread reads its window from the tile; w as in naive.
//                   Each x is loaded from global about once per block (halo overhead
//                   (taps - 1) / kThreads). The filter is walked in chunks of kChunk taps, so
//                   shared memory stays fixed for any K.
//   3 conv1d_smem_reg the conv1d_smem tile, stored in FP32 with conflict-free padding; each
//                   thread computes kRegOutputs consecutive outputs from a sliding register
//                   window: per tap one shared load, one w load, kRegOutputs FMAs.
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

// conv1d_smem filter chunk: a tile of kThreads + kChunk - 1 elements (5 KB in FP32).
constexpr int kChunk = 1024;
// conv1d_smem_reg outputs per thread (a power of two keeps pad() a shift).
constexpr int kRegOutputs = 8;

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

// Every thread of the block must reach both barriers, including threads past the last
// output (i >= m) in the final block: they still load their share of the tile and only
// skip the store. Tile slots past the end of x are zero-filled, so no thread reads
// unwritten shared memory.
template <typename T>
__global__ void conv1d_smem(const T *__restrict__ x, const T *__restrict__ w, T *__restrict__ out,
                            int64_t n, int64_t m, int k) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T *tile = reinterpret_cast<T *>(smem_raw);
  const int64_t base = int64_t(blockIdx.x) * blockDim.x;
  const int64_t i = base + threadIdx.x;

  float acc = 0.0f;
  for (int j0 = 0; j0 < k; j0 += kChunk) {
    const int taps = min(kChunk, k - j0);
    const int width = blockDim.x + taps - 1;
    const int64_t start = base + j0;  // x index of tile[0]
    for (int t = threadIdx.x; t < width; t += blockDim.x)
      tile[t] = start + t < n ? x[start + t] : T(0.0f);
    __syncthreads();

    const T *xi = tile + threadIdx.x;
    for (int j = 0; j < taps; ++j)
      acc = fmaf(static_cast<float>(xi[j]), static_cast<float>(w[j0 + j]), acc);
    __syncthreads();  // the next chunk overwrites the tile
  }
  if (i < m)
    out[i] = static_cast<T>(acc);
}


// Register-blocked shared tile. Thread t owns R consecutive outputs o = base + t*R + r and
// keeps a sliding window win[] of R x values in registers. Per tap it loads one new x from
// shared memory and one w (a broadcast global load), then does R FMAs, so shared loads per
// FMA fall from 1 (cuda_smem) to 1/R and w loads to 1/R.
//
// The tap loop runs in groups of R with every window index a compile-time constant: at step
// s of a group, output r needs window slot (r + s) % R, and slot s is refilled after use, so
// the window rotates without register moves. Fewer than R leftover taps read the tile
// directly.
//
// The tile is stored in FP32, converted once on load rather than once per tap. Lanes read
// the tile R elements apart, which is an R-way bank conflict for R = 8; padding every R
// elements with one empty slot (pad(idx) = idx + idx / R) makes the stride R + 1, odd, so
// a warp's tap read is conflict-free. As in conv1d_smem, every thread reaches every
// barrier and slots past the end of x hold zero.
template <int R>
__device__ __forceinline__ int pad(int idx) {
  return idx + idx / R;
}

template <typename T, int R>
__global__ void conv1d_smem_reg(const T *__restrict__ x, const T *__restrict__ w, T *__restrict__ out,
                                int64_t n, int64_t m, int k) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  float *tile = reinterpret_cast<float *>(smem_raw);
  const int span = blockDim.x * R;  // outputs per block
  const int64_t base = int64_t(blockIdx.x) * span;
  const int t0 = threadIdx.x * R;  // this thread's first output, relative to base

  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r)
    acc[r] = 0.0f;

  for (int j0 = 0; j0 < k; j0 += kChunk) {
    const int taps = min(kChunk, k - j0);
    // span + taps - 1 elements cover every window; the last group's refill reads one more.
    const int width = span + taps;
    const int64_t start = base + j0;  // x index of tile[0]
    for (int idx = threadIdx.x; idx < width; idx += blockDim.x)
      tile[pad<R>(idx)] = start + idx < n ? static_cast<float>(x[start + idx]) : 0.0f;
    __syncthreads();

    float win[R];
#pragma unroll
    for (int q = 0; q < R; ++q)
      win[q] = tile[pad<R>(t0 + q)];

    const T *wc = w + j0;
    int j = 0;
    for (; j + R <= taps; j += R) {
#pragma unroll
      for (int s = 0; s < R; ++s) {
        const float wj = static_cast<float>(wc[j + s]);
#pragma unroll
        for (int r = 0; r < R; ++r)
          acc[r] = fmaf(win[(r + s) % R], wj, acc[r]);
        win[s] = tile[pad<R>(t0 + j + R + s)];
      }
    }
    for (; j < taps; ++j) {  // fewer than R leftover taps
      const float wj = static_cast<float>(wc[j]);
#pragma unroll
      for (int r = 0; r < R; ++r)
        acc[r] = fmaf(tile[pad<R>(t0 + r + j)], wj, acc[r]);
    }
    __syncthreads();  // the next chunk overwrites the tile
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int64_t o = base + t0 + r;
    if (o < m)
      out[o] = static_cast<T>(acc[r]);
  }
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
    const int k32 = static_cast<int>(k);
    if (variant == 0) {
      conv1d_naive<scalar_t><<<blocks, kThreads, 0, stream>>>(xp, w.data_ptr<scalar_t>(), op, m, k32);
    } else if (variant == 1) {
      conv1d_const<scalar_t><<<blocks, kThreads, 0, stream>>>(xp, op, m, k32);
    } else if (variant == 2) {
      const size_t smem = (kThreads + kChunk - 1) * sizeof(scalar_t);
      conv1d_smem<scalar_t><<<blocks, kThreads, smem, stream>>>(xp, w.data_ptr<scalar_t>(), op, n,
                                                                m, k32);
    } else {
      constexpr int R = kRegOutputs;
      // FP32 tile of span + kChunk elements after padding, plus one slot for the last pad.
      const int width = kThreads * R + kChunk;
      const size_t smem = (width + width / R + 1) * sizeof(float);
      conv1d_smem_reg<scalar_t, R><<<lab::grid_blocks(m, kThreads * R), kThreads, smem, stream>>>(
          xp, w.data_ptr<scalar_t>(), op, n, m, k32);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
