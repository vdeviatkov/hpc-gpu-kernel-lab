// Variants: 2-D row-major and column-major thread mappings, a flat scalar copy, 16-byte
// packs, packs with 2 or 4 loads in flight per thread, a one-wave grid-stride loop, and
// packs with streaming cache hints.
//
// A copy never interprets its data, so the kernels move unsigned integers of the element's
// width: one instantiation serves FP32 and one serves FP16 and BF16, and bit patterns
// (NaN payloads, -0.0, subnormals) pass through unchanged by construction.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "lab/launch.cuh"
#include "lab/vector.cuh"

namespace {

constexpr int kWarp = 32;
// gridDim.y and gridDim.z are limited to 65535; larger extents are covered by looping.
constexpr int64_t kMaxGridY = 65535;

// threadIdx.x walks along a row: a warp reads and writes 32 consecutive elements.
template <typename T>
__global__ void rows_kernel(const T *__restrict__ a, T *__restrict__ out, int64_t m, int64_t n) {
  const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= n)
    return;
  for (int64_t row = int64_t(blockIdx.y) * blockDim.y + threadIdx.y; row < m;
       row += int64_t(gridDim.y) * blockDim.y)
    out[row * n + col] = a[row * n + col];
}

// threadIdx.x walks down a column: a warp's 32 elements are n apart, so for n >= 32 FP32
// (64 FP16) every lane touches a different 32-byte sector. Deliberately uncoalesced.
template <typename T>
__global__ void columns_kernel(const T *__restrict__ a, T *__restrict__ out, int64_t m, int64_t n) {
  const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= m)
    return;
  for (int64_t col = int64_t(blockIdx.y) * blockDim.y + threadIdx.y; col < n;
       col += int64_t(gridDim.y) * blockDim.y)
    out[row * n + col] = a[row * n + col];
}

// The matrix is contiguous, so it is also one vector of m * n elements.
template <typename T>
__global__ void flat_kernel(const T *__restrict__ a, T *__restrict__ out, int64_t total) {
  const int64_t i = lab::global_index();
  if (i < total)
    out[i] = a[i];
}

// The last total % kSize elements, which do not fill a pack: one per thread among the first
// few threads of the grid. Every pack launch has at least one full block, and a block has
// more threads than a pack has elements.
template <typename T>
__device__ __forceinline__ void copy_tail(const T *__restrict__ a, T *__restrict__ out,
                                          int64_t total) {
  constexpr int64_t K = lab::Pack<T>::kSize;
  const int64_t i = total / K * K + lab::global_index();
  if (i < total)
    out[i] = a[i];
}

template <typename T>
__global__ void pack_kernel(const T *__restrict__ a, T *__restrict__ out, int64_t packs,
                            int64_t total) {
  const int64_t i = lab::global_index();
  if (i < packs)
    lab::store_pack(out, i, lab::load_pack(a, i));
  copy_tail(a, out, total);
}

// Each thread copies U packs blockDim.x apart, so every warp access stays contiguous, and
// issues all U loads before the first store: U * 16 bytes in flight per thread.
template <typename T, int U>
__global__ void ilp_kernel(const T *__restrict__ a, T *__restrict__ out, int64_t packs,
                           int64_t total) {
  const int64_t base = int64_t(blockIdx.x) * blockDim.x * U + threadIdx.x;
  lab::Pack<T> values[U];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int64_t p = base + int64_t(u) * blockDim.x;
    if (p < packs)
      values[u] = lab::load_pack(a, p);
  }
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int64_t p = base + int64_t(u) * blockDim.x;
    if (p < packs)
      lab::store_pack(out, p, values[u]);
  }
  copy_tail(a, out, total);
}

// Launched with one wave of resident blocks; each thread loops over the packs.
template <typename T>
__global__ void grid_stride_kernel(const T *__restrict__ a, T *__restrict__ out, int64_t packs,
                                   int64_t total) {
  for (int64_t i = lab::global_index(); i < packs; i += lab::grid_threads())
    lab::store_pack(out, i, lab::load_pack(a, i));
  copy_tail(a, out, total);
}

// __ldcs/__stcs (ld.global.cs / st.global.cs) mark lines evict-first, so data touched once
// should not displace other L2 contents. The intrinsics take uint4, a 16-byte pack.
template <typename T>
__global__ void streaming_kernel(const T *__restrict__ a, T *__restrict__ out, int64_t packs,
                                 int64_t total) {
  const int64_t i = lab::global_index();
  if (i < packs)
    __stcs(reinterpret_cast<uint4 *>(out) + i, __ldcs(reinterpret_cast<const uint4 *>(a) + i));
  copy_tail(a, out, total);
}

template <typename T>
void launch(const T *a, T *out, int64_t m, int64_t n, int64_t variant, int threads,
            cudaStream_t stream) {
  const int64_t total = m * n;
  // Packs need 16-byte aligned input and output; otherwise use the flat scalar copy.
  if (variant >= 3 && !lab::pack_aligned({a, out}))
    variant = 2;
  const int64_t packs = total / lab::Pack<T>::kSize;
  // At least one block, which also covers the tail.
  const auto pack_blocks = [&](int64_t per_thread) {
    return lab::grid_blocks(std::max<int64_t>((packs + per_thread - 1) / per_thread, 1), threads);
  };
  const dim3 block2d(kWarp, threads / kWarp);
  switch (variant) {
  case 0: {
    const dim3 grid(lab::grid_blocks(n, kWarp),
                    std::min<int64_t>((m + block2d.y - 1) / block2d.y, kMaxGridY));
    rows_kernel<<<grid, block2d, 0, stream>>>(a, out, m, n);
    break;
  }
  case 1: {
    const dim3 grid(lab::grid_blocks(m, kWarp),
                    std::min<int64_t>((n + block2d.y - 1) / block2d.y, kMaxGridY));
    columns_kernel<<<grid, block2d, 0, stream>>>(a, out, m, n);
    break;
  }
  case 2:
    flat_kernel<<<lab::grid_blocks(total, threads), threads, 0, stream>>>(a, out, total);
    break;
  case 3:
    pack_kernel<<<pack_blocks(1), threads, 0, stream>>>(a, out, packs, total);
    break;
  case 4:
    ilp_kernel<T, 2><<<pack_blocks(2), threads, 0, stream>>>(a, out, packs, total);
    break;
  case 5:
    ilp_kernel<T, 4><<<pack_blocks(4), threads, 0, stream>>>(a, out, packs, total);
    break;
  case 6: {
    int per_sm = 0;
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, grid_stride_kernel<T>, threads, 0));
    const int64_t wave =
        int64_t(per_sm) * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int blocks =
        lab::grid_blocks(std::max<int64_t>(packs, 1), threads, std::max<int64_t>(wave, 1));
    grid_stride_kernel<<<blocks, threads, 0, stream>>>(a, out, packs, total);
    break;
  }
  default:
    streaming_kernel<<<pack_blocks(1), threads, 0, stream>>>(a, out, packs, total);
  }
}

} // namespace

void launch_matrix_copy(const at::Tensor &a, at::Tensor out, int64_t variant, int64_t threads) {
  const c10::cuda::CUDAGuard guard(a.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const int64_t m = a.size(0);
  const int64_t n = a.size(1);
  if (!a.numel())
    return;
  if (a.element_size() == 4)
    launch(static_cast<const uint32_t *>(a.data_ptr()), static_cast<uint32_t *>(out.data_ptr()), m,
           n, variant, static_cast<int>(threads), stream);
  else
    launch(static_cast<const uint16_t *>(a.data_ptr()), static_cast<uint16_t *>(out.data_ptr()), m,
           n, variant, static_cast<int>(threads), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
