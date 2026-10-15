#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <numeric>

enum Variant { kPixel = 0, kVec = 1, kSmem = 2, kPlanar = 3 };

// The source's formula with its exact coefficients, always computed in FP32 (float literals:
// double literals would promote the whole expression to FP64, which is both off-contract and
// many times slower on this GPU). nvcc fuses the multiply-adds into FFMA instructions, so the
// last bit can differ from PyTorch's separately rounded products; tests use tolerances.
template <typename T> __device__ __forceinline__ T gray(T r, T g, T b) {
  return T(0.299f * static_cast<float>(r) + 0.587f * static_cast<float>(g) +
           0.114f * static_cast<float>(b));
}

// One thread per pixel on a 2-D grid: x indexes the column, y the row. Pixel (row, col) of a
// contiguous (H, W, 3) image starts at element (row * W + col) * 3. Neighbouring lanes are one
// pixel (3 elements, 12 bytes in FP32) apart, so each of the three loads is a stride-3 access.
template <typename T>
__global__ void rgb_to_grayscale_pixel(const T *__restrict__ image, T *__restrict__ out, int64_t h,
                                       int64_t w) {
  const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= h || col >= w)
    return;
  const int64_t pixel = row * w + col;
  const T *rgb = image + pixel * 3;
  out[pixel] = gray(rgb[0], rgb[1], rgb[2]);
}

namespace {
constexpr size_t VectorBytes = 16;

template <typename T, size_t ChannelsPerPixel> struct Pixel {
  static constexpr size_t ChannelsCount = ChannelsPerPixel;

  T channel[ChannelsPerPixel];
};

// The smallest run of whole pixels that is also a whole number of 16-byte packs:
// lcm(pixel size, 16) bytes. For RGB that is 48 bytes, i.e. 4 FP32 or 8 FP16/BF16 pixels in
// three packs. alignas lets the compiler move it with three 128-bit loads.
template <typename T, size_t ChannelsPerPixel> struct Vector {
  static constexpr size_t VectorSize = std::lcm(sizeof(Pixel<T, ChannelsPerPixel>), VectorBytes);
  static constexpr size_t PixelsPerVector = VectorSize / sizeof(Pixel<T, ChannelsPerPixel>);
  static constexpr size_t ChannelsPerVector = VectorSize / sizeof(T);

  alignas(VectorBytes) Pixel<T, ChannelsPerPixel> pixels[PixelsPerVector];
};

// The gray values of one Vector: PixelsPerVector elements, one 16-byte pack for RGB.
template <typename T, size_t Size> struct GrayscaleVector {
  static constexpr size_t PixelsPerVector = Size;

  alignas(VectorBytes) T pixels[Size];
};

// Reading T arrays through these structs is the usual CUDA vector-access idiom; see the
// aliasing note in lab/vector.cuh. The launcher checks 16-byte alignment first.
template <typename T, size_t ChannelsPerPixel>
__device__ __forceinline__ Vector<T, ChannelsPerPixel> loadVector(const T *__restrict__ ptr,
                                                                  int64_t i) {
  return reinterpret_cast<const Vector<T, ChannelsPerPixel> *>(ptr)[i];
}

template <typename T, size_t Size>
__device__ __forceinline__ void storeGrayscaleVector(T *__restrict__ ptr, int64_t i,
                                                     const GrayscaleVector<T, Size> &value) {
  reinterpret_cast<GrayscaleVector<T, Size> *>(ptr)[i] = value;
}
} // namespace

// One thread per Vector of pixels, on a 1-D grid over the whole image. Rows do not matter:
// the image is contiguous, so pixel group v (pixels v*k .. v*k + k - 1) is input Vector v and
// output GrayscaleVector v for any width; no width alignment is needed. The thread just past
// the last full group handles the remaining pixels % k pixels with scalar code.
template <typename T>
__global__ void rgb_to_grayscale_vec(const T *__restrict__ image, T *__restrict__ out,
                                     int64_t pixels) {
  using SrcVector = Vector<T, 3>;
  using DstVector = GrayscaleVector<T, SrcVector::PixelsPerVector>;
  constexpr int64_t k = SrcVector::PixelsPerVector;

  const int64_t v = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t vectors = pixels / k;
  if (v < vectors) {
    const SrcVector src = loadVector<T, 3>(image, v);
    DstVector result;
#pragma unroll
    for (int64_t i = 0; i < k; ++i) {
      result.pixels[i] =
          gray(src.pixels[i].channel[0], src.pixels[i].channel[1], src.pixels[i].channel[2]);
    }
    storeGrayscaleVector(out, v, result);
  } else if (v == vectors) {
#pragma nounroll
    for (int64_t p = vectors * k; p < pixels; ++p) {
      out[p] = gray(image[p * 3], image[p * 3 + 1], image[p * 3 + 2]);
    }
  }
}

// ---- cuda_smem: stage each tile's interleaved input through shared memory ----------------
//
// A block of kTileThreads threads handles a tile of kTileThreads pixels, one per thread. The
// tile's input is the contiguous run image[first * 3, (first + count) * 3): 3 KB in FP32,
// 1.5 KB in FP16/BF16, a whole number of 16-byte packs that starts 16-byte aligned (given an
// aligned image). Phase 1 copies it into shared memory with coalesced 16-byte loads; phase 2
// lets thread t read its pixel at tile[3t .. 3t + 2]. In FP32 that stride of 3 words is free
// of bank conflicts: 3 and the 32 banks share no factor, so the 32 lanes of a warp hit 32
// different banks. With 2-byte types two lanes can share a 4-byte word (a broadcast, no
// conflict) but 32 lanes span 48 words, so some banks see two words: up to 2-way conflicts.
constexpr int kTileThreads = 256;

template <typename T>
__global__ void rgb_to_grayscale_smem(const T *__restrict__ image, T *__restrict__ out,
                                      int64_t pixels) {
  constexpr int tile_elements = kTileThreads * 3;
  constexpr int tile_bytes = tile_elements * static_cast<int>(sizeof(T));
  static_assert(tile_bytes % VectorBytes == 0, "a tile must be whole 16-byte packs");
  constexpr int elements_per_pack = VectorBytes / sizeof(T);
  struct alignas(VectorBytes) TilePack {
    T v[elements_per_pack];
  };
  // A raw byte buffer: shared variables cannot have constructors, which c10::Half has.
  __shared__ __align__(16) unsigned char raw[tile_bytes];
  T *tile = reinterpret_cast<T *>(raw);

  const int64_t first = int64_t(blockIdx.x) * kTileThreads;
  const int64_t remaining = pixels - first;
  const int count = remaining < kTileThreads ? static_cast<int>(remaining) : kTileThreads;
  const T *src = image + first * 3;

  // Phase 1: global -> shared. Full tiles move as 16-byte packs; the last, partial tile
  // (fewer than kTileThreads pixels) moves element by element.
  if (count == kTileThreads) {
    for (int pack = threadIdx.x; pack < tile_bytes / VectorBytes; pack += blockDim.x)
      reinterpret_cast<TilePack *>(tile)[pack] = reinterpret_cast<const TilePack *>(src)[pack];
  } else {
    for (int e = threadIdx.x; e < count * 3; e += blockDim.x)
      tile[e] = src[e];
  }
  __syncthreads(); // every thread's pixel may have been loaded by another thread

  // Phase 2: one pixel per thread from shared memory, one coalesced store.
  const int t = threadIdx.x;
  if (t < count)
    out[first + t] = gray(tile[3 * t], tile[3 * t + 1], tile[3 * t + 2]);
}

// ---- cuda_planar: convert to planar (R, G and B as separate arrays), then compute ---------
//
// Answers "would a planar layout help, and is converting worth it?". The api receives
// interleaved RGB, so this backend pays for the conversion: deinterleave_rgb reads the image
// (stride-3 loads, like cuda_pixel) and writes three planes; rgb_to_grayscale_planar then reads
// R[p], G[p], B[p], each a perfectly coalesced load. A profiler shows the two kernels
// separately: the second is what the computation would cost if the data were already planar.
template <typename T>
__global__ void deinterleave_rgb(const T *__restrict__ image, T *__restrict__ planes,
                                 int64_t pixels) {
  const int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= pixels)
    return;
  planes[p] = image[p * 3];
  planes[pixels + p] = image[p * 3 + 1];
  planes[2 * pixels + p] = image[p * 3 + 2];
}

template <typename T>
__global__ void rgb_to_grayscale_planar(const T *__restrict__ r, const T *__restrict__ g,
                                        const T *__restrict__ b, T *__restrict__ out,
                                        int64_t pixels) {
  const int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= pixels)
    return;
  out[p] = gray(r[p], g[p], b[p]);
}

static int blocks_for(int64_t items, int threads) {
  const int64_t blocks = (items + threads - 1) / threads;
  TORCH_CHECK(blocks <= 2147483647LL, "rgb_to_grayscale: image exceeds the launch grid limit");
  return static_cast<int>(blocks);
}

void launch_rgb_to_grayscale(const at::Tensor &image, at::Tensor out, int64_t variant) {
  const c10::cuda::CUDAGuard guard(image.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(image.get_device()).stream();
  const int64_t h = image.size(0);
  const int64_t w = image.size(1);
  if (h == 0 || w == 0)
    return; // A zero-sized grid would be a launch error.
  // Vector loads need a 16-byte aligned image (a fresh tensor always is; a row slice of a
  // larger image may not be); the output is freshly allocated. Otherwise: one pixel per thread.
  // cuda_smem's 16-byte tile loads need the same alignment.
  if ((variant == kVec || variant == kSmem) &&
      reinterpret_cast<uintptr_t>(image.data_ptr()) % VectorBytes != 0)
    variant = kPixel;
  const int64_t pixels = h * w;
  // Temporary R, G, B planes for cuda_planar, from PyTorch's stream-ordered allocator.
  const at::Tensor planes =
      variant == kPlanar ? at::empty({3, pixels}, image.options()) : at::Tensor();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, image.scalar_type(), "rgb_to_gray", [&] {
        const auto *src = image.data_ptr<scalar_t>();
        auto *dst = out.data_ptr<scalar_t>();
        if (variant == kVec) {
          constexpr int threads = 256;
          constexpr int64_t k = Vector<scalar_t, 3>::PixelsPerVector;
          rgb_to_grayscale_vec<<<blocks_for(h * w / k + 1, threads), threads, 0, stream>>>(src, dst,
                                                                                           h * w);
        } else if (variant == kSmem) {
          rgb_to_grayscale_smem<<<blocks_for(pixels, kTileThreads), kTileThreads, 0, stream>>>(
              src, dst, pixels);
        } else if (variant == kPlanar) {
          constexpr int threads = 256;
          auto *r = planes.data_ptr<scalar_t>();
          deinterleave_rgb<<<blocks_for(pixels, threads), threads, 0, stream>>>(src, r, pixels);
          rgb_to_grayscale_planar<<<blocks_for(pixels, threads), threads, 0, stream>>>(
              r, r + pixels, r + 2 * pixels, dst, pixels);
        } else {
          // 32 x 8 threads: one warp covers 32 consecutive pixels of a row.
          const dim3 block(32, 8);
          const int64_t grid_y = (h + block.y - 1) / block.y;
          TORCH_CHECK(grid_y <= 65535, "rgb_to_grayscale: image too tall for the 2-D grid");
          const dim3 grid(blocks_for(w, block.x), static_cast<unsigned>(grid_y));
          rgb_to_grayscale_pixel<<<grid, block, 0, stream>>>(src, dst, h, w);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
