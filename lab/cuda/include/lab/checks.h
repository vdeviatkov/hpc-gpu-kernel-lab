// Tensor checks for the C++ boundary (bindings.cpp). The Python api validates first;
// these repeat the checks so a direct call into the extension cannot launch on bad input.
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cstdint>

namespace lab {

// A contiguous, strided, 1-D CUDA tensor.
inline void check_cuda_vector(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.layout() == c10::kStrided && t.dim() == 1 && t.is_contiguous(), name,
              ": expected a contiguous 1-D CUDA tensor");
}

// Same shape, dtype and device as `reference`.
inline void check_same(const at::Tensor &t, const at::Tensor &reference, const char *name) {
  TORCH_CHECK(t.sizes() == reference.sizes() && t.scalar_type() == reference.scalar_type() &&
                  t.device() == reference.device(),
              name, ": shape, dtype and device must match");
}

// FP32, FP16 or BF16: the dtypes the lab's floating-point kernels are built for.
inline void check_float_dtype(const at::Tensor &t, const char *name) {
  const auto type = t.scalar_type();
  TORCH_CHECK(type == at::kFloat || type == at::kHalf || type == at::kBFloat16, name,
              ": supported dtypes are fp32, fp16 and bf16");
}

// No autograd and no lazily-resolved negation or conjugation views.
inline void check_plain(const at::Tensor &t, const char *name) {
  TORCH_CHECK(!t.requires_grad() && !t.is_neg() && !t.is_conj(), name,
              ": autograd and unresolved negative/conjugate views are unsupported");
}

// `out` shares no bytes with `in`.
inline void check_no_overlap(const at::Tensor &out, const at::Tensor &in, const char *name) {
  const auto out_begin = reinterpret_cast<std::uintptr_t>(out.data_ptr());
  const auto in_begin = reinterpret_cast<std::uintptr_t>(in.data_ptr());
  TORCH_CHECK(!(out_begin < in_begin + in.nbytes() && in_begin < out_begin + out.nbytes()), name,
              ": output overlaps an input");
}

} // namespace lab
