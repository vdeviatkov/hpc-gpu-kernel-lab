#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_interleave(const at::Tensor &a, const at::Tensor &b, at::Tensor out, int64_t variant,
                       int64_t threads);

void interleave_out(const at::Tensor &a, const at::Tensor &b, at::Tensor out, int64_t variant,
                    int64_t threads) {
  TORCH_CHECK(variant >= 0 && variant <= 3, "interleave: invalid variant");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "interleave: invalid block size");
  for (const auto &x : {a, b, out}) {
    lab::check_cuda_vector(x, "interleave");
    lab::check_plain(x, "interleave");
  }
  lab::check_same(b, a, "interleave");
  lab::check_float_dtype(a, "interleave");
  TORCH_CHECK(out.numel() == 2 * a.numel() && out.scalar_type() == a.scalar_type() &&
                  out.device() == a.device(),
              "interleave: output must hold 2N elements of the input dtype on the same device");
  lab::check_no_overlap(out, a, "interleave");
  lab::check_no_overlap(out, b, "interleave");
  launch_interleave(a, b, out, variant, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("interleave_out", &interleave_out, "04 · Interleave Arrays into a separate output");
}
