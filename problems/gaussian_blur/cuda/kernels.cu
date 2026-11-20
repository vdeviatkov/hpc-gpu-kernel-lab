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

template <typename T>
__global__ void gaussian_blur_direct(const T *__restrict__ image, const T *__restrict__ kernel,
                                     T *__restrict__ out, int h, int w, int kh, int kw) {
  const int col = blockIdx.x * kBlockX + threadIdx.x;
  const int row = blockIdx.y * kBlockY + threadIdx.y;
  if (row >= h || col >= w)
    return;
  const int top = row - kh / 2;
  const int left = col - kw / 2;
  // Clip the window to the image instead of testing every tap: padded taps contribute 0,
  // so skipping them is exact.
  const int m0 = max(0, -top), m1 = min(kh, h - top);
  const int n0 = max(0, -left), n1 = min(kw, w - left);
  float acc = 0.0f;
  for (int m = m0; m < m1; ++m) {
    // Offset by the window's first column per tap, not in the row pointer: `left` is
    // negative at the left border and a pointer before the image is undefined.
    const T *image_row = image + int64_t(top + m) * w;
    const T *kernel_row = kernel + m * kw;
    for (int n = n0; n < n1; ++n)
      acc = fmaf(static_cast<float>(image_row[left + n]), static_cast<float>(kernel_row[n]), acc);
  }
  out[int64_t(row) * w + col] = static_cast<T>(acc);
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
  TORCH_CHECK(variant == 0, "gaussian_blur: invalid variant");

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, image.scalar_type(), "gaussian_blur", [&] {
    gaussian_blur_direct<scalar_t><<<grid, block, 0, stream>>>(
        image.data_ptr<scalar_t>(), kernel.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
        static_cast<int>(h), static_cast<int>(w), static_cast<int>(kh), static_cast<int>(kw));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
