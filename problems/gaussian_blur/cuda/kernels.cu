// 10 · Gaussian Blur: CUDA kernels.
//
// Same-size 2-D cross-correlation with zero padding: out[i, j] = sum_{m, n}
// image[i + m - KH/2, j + n - KW/2] * kernel[m, n], products accumulated in FP32 and rounded
// once to the input dtype. Variants:
//
//   0 gaussian_blur_direct  one output pixel per thread, the full KH x KW loop, image and
//                           kernel read from global memory. A warp spans 32 consecutive
//                           columns of one row, so each tap is one coalesced (but
//                           column-misaligned) image load plus one broadcast kernel load;
//                           L1 absorbs the KH*KW-fold reuse, so LSU/L1 requests per FMA, not
//                           DRAM, are the expected limit (conv1d cuda_naive's plateau).
//   1 gaussian_blur_const   the same loop with the kernel converted to FP32 and staged in
//                           __constant__ memory (at most kMaxConstTaps taps). The weight index
//                           is warp-uniform, so each weight is loaded once per warp into a
//                           uniform register and the kernel loads leave the LSU/L1 pipe:
//                           about half the requests per FMA (conv1d cuda_const).
//
// Both run the full KH x KW window with predicated image loads; see blur_pixel.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {

// 32 x 8 threads: a warp is one row segment of 32 columns, so image loads and output stores
// coalesce along the row.
constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

constexpr int kMaxConstTaps = 16384; // 64 KiB of FP32: the whole __constant__ bank
__constant__ float c_w[kMaxConstTaps];

// One output pixel. The tap loops run over the full KH x KW window with bounds that are the
// same for every lane, and padding is applied by predicating the image load, not by
// clipping the loop per thread: the weight index m * kw + n is then provably warp-uniform,
// which is what lets the __constant__ variant load each weight once per warp into a uniform
// register (LDCU). A per-thread clipped loop (start max(0, -left)) would make the index
// lane-dependent and turn those into per-lane constant loads.
template <typename T, typename W>
__device__ __forceinline__ void blur_pixel(const T *__restrict__ image, const W *__restrict__ weights,
                                           T *__restrict__ out, int h, int w, int kh, int kw) {
  const int col = blockIdx.x * kBlockX + threadIdx.x;
  const int row = blockIdx.y * kBlockY + threadIdx.y;
  if (row >= h || col >= w)
    return;
  const int top = row - kh / 2;
  const int left = col - kw / 2;
  float acc = 0.0f;
  for (int m = 0; m < kh; ++m) {
    const int r = top + m;
    const bool row_in = r >= 0 && r < h;
    // Row offset only when the row exists: a pointer outside the image is undefined.
    const T *image_row = image + (row_in ? int64_t(r) * w : 0);
    for (int n = 0; n < kw; ++n) {
      const int c = left + n;
      const float x = (row_in && c >= 0 && c < w) ? static_cast<float>(image_row[c]) : 0.0f;
      acc = fmaf(x, static_cast<float>(weights[m * kw + n]), acc);
    }
  }
  out[int64_t(row) * w + col] = static_cast<T>(acc);
}

template <typename T>
__global__ void gaussian_blur_direct(const T *__restrict__ image, const T *__restrict__ kernel,
                                     T *__restrict__ out, int h, int w, int kh, int kw) {
  blur_pixel(image, kernel, out, h, w, kh, kw);
}

template <typename T>
__global__ void gaussian_blur_const(const T *__restrict__ image, T *__restrict__ out, int h, int w,
                                    int kh, int kw) {
  blur_pixel(image, c_w, out, h, w, kh, kw);
}

} // namespace

void launch_gaussian_blur(const at::Tensor &image, const at::Tensor &kernel, at::Tensor out,
                          int64_t variant) {
  const c10::cuda::CUDAGuard guard(image.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(image.get_device()).stream();
  const int64_t h = image.size(0), w = image.size(1);
  const int64_t kh = kernel.size(0), kw = kernel.size(1);
  // h - top and w - left in the kernel reach h + kh/2 and w + kw/2.
  TORCH_CHECK(h + kh <= INT_MAX && w + kw <= INT_MAX && kh * kw <= INT_MAX,
              "gaussian_blur: dimensions exceed INT_MAX");
  const int64_t grid_y = (h + kBlockY - 1) / kBlockY;
  TORCH_CHECK(grid_y <= 65535, "gaussian_blur: more than ", 65535 * kBlockY, " rows");
  const dim3 block(kBlockX, kBlockY);
  const dim3 grid(static_cast<unsigned>((w + kBlockX - 1) / kBlockX), static_cast<unsigned>(grid_y));
  TORCH_CHECK(variant == 0 || variant == 1, "gaussian_blur: invalid variant");

  if (variant == 1) {
    TORCH_CHECK(kh * kw <= kMaxConstTaps, "gaussian_blur: cuda_direct_const holds at most ",
                kMaxConstTaps, " kernel taps in __constant__ memory");
    // Converted on the current stream (no copy for contiguous FP32); the caching allocator
    // keeps the temporary alive until the stream has consumed it. Stream-ordered, so a later
    // call's upload cannot overwrite the weights before this launch has read them.
    const at::Tensor w32 = kernel.to(at::kFloat).contiguous();
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_w, w32.data_ptr<float>(), kh * kw * sizeof(float), 0,
                                           cudaMemcpyDeviceToDevice, stream));
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, image.scalar_type(), "gaussian_blur", [&] {
    const auto *ip = image.data_ptr<scalar_t>();
    auto *op = out.data_ptr<scalar_t>();
    const int h32 = static_cast<int>(h), w32 = static_cast<int>(w);
    const int kh32 = static_cast<int>(kh), kw32 = static_cast<int>(kw);
    if (variant == 0)
      gaussian_blur_direct<scalar_t>
          <<<grid, block, 0, stream>>>(ip, kernel.data_ptr<scalar_t>(), op, h32, w32, kh32, kw32);
    else
      gaussian_blur_const<scalar_t><<<grid, block, 0, stream>>>(ip, op, h32, w32, kh32, kw32);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t max_const_taps() { return kMaxConstTaps; }
