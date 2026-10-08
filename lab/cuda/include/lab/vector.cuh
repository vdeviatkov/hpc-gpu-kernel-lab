// 16-byte vector access shared by every problem's kernels.
//
//   #include "lab/vector.cuh"
//   const lab::Pack<T> v = lab::load_pack(x, i);   // elements 4i..4i+3 (FP32) or 8i..8i+7 (FP16)
//   lab::store_pack(out, i, v);
//
// Callers must check alignment on the host with lab::pack_aligned() and fall back to
// scalar code when it fails; nothing here assumes alignment it cannot see.
#pragma once

#include <cstdint>
#include <initializer_list>

namespace lab {

// The widest single global-memory access a thread can make (LDG.E.128 / STG.E.128).
constexpr int kVectorBytes = 16;

// kSize consecutive elements of T in one 16-byte aligned object: four FP32 or eight
// FP16/BF16. The alignas is what lets the compiler emit one 128-bit load or store.
template <typename T> struct alignas(kVectorBytes) Pack {
  static constexpr int kSize = kVectorBytes / static_cast<int>(sizeof(T));
  T v[kSize];
};

// Reading T arrays through Pack<T> pointers is technically undefined behavior in ISO C++
// (strict aliasing: no Pack<T> object exists at those addresses). It is nevertheless the
// standard CUDA idiom for vector access, the same pattern as NVIDIA's float4 casts and
// PyTorch's aligned_vector, and nvcc compiles it as intended: one LDG.E.128/STG.E.128 per
// pack, verified in SASS. A memcpy into Pack<T> would be well-defined, but must be
// re-verified in SASS to still produce 128-bit accesses.
template <typename T> __device__ __forceinline__ Pack<T> load_pack(const T *p, int64_t i) {
  return reinterpret_cast<const Pack<T> *>(p)[i];
}

template <typename T>
__device__ __forceinline__ void store_pack(T *p, int64_t i, const Pack<T> &value) {
  reinterpret_cast<Pack<T> *>(p)[i] = value;
}

// Host check: true when every pointer is 16-byte aligned, so Pack access is legal.
inline bool pack_aligned(std::initializer_list<const void *> pointers) {
  std::uintptr_t bits = 0;
  for (const void *p : pointers)
    bits |= reinterpret_cast<std::uintptr_t>(p);
  return bits % kVectorBytes == 0;
}

} // namespace lab
