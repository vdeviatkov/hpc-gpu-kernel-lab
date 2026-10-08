// Thread indexing and grid sizing shared by every problem's kernels.
//
//   #include "lab/launch.cuh"
//   const int64_t i = lab::global_index();                 // in a kernel
//   kernel<<<lab::grid_blocks(n, threads), threads, 0, stream>>>(...);   // on the host
#pragma once

#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>

namespace lab {

// This thread's index in a 1-D grid, in 64 bits so N > 2^31 cannot overflow.
__device__ __forceinline__ int64_t global_index() {
  return int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
}

// Total threads in a 1-D grid: the step of a grid-stride loop.
__device__ __forceinline__ int64_t grid_threads() { return int64_t(gridDim.x) * blockDim.x; }

// Blocks needed to give each of `items` work items one thread. With max_blocks > 0 the
// grid is capped, for grid-stride loops that cover the rest by iterating.
inline int grid_blocks(int64_t items, int threads, int64_t max_blocks = 0) {
  const int64_t needed = (items + threads - 1) / threads;
  if (max_blocks > 0)
    return static_cast<int>(std::min(needed, max_blocks));
  TORCH_CHECK(needed <= 2147483647LL, "input exceeds the 1-D launch grid limit");
  return static_cast<int>(needed);
}

} // namespace lab
