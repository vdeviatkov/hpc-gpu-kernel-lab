#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_add(const at::Tensor &a, const at::Tensor &b, at::Tensor &out, int variant,
                int threads);

void add_out(const at::Tensor &a, const at::Tensor &b, at::Tensor out, int variant, int threads) {
  TORCH_CHECK(variant >= 0 && variant <= 3, "vector_add: invalid variant");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "vector_add: invalid block size");
  for (const auto &x : {a, b, out}) {
    lab::check_cuda_vector(x, "vector_add");
    lab::check_same(x, a, "vector_add");
    lab::check_plain(x, "vector_add");
  }
  lab::check_float_dtype(a, "vector_add");
  if (a.numel() == 0)
    return;
  lab::check_no_overlap(out, a, "vector_add");
  lab::check_no_overlap(out, b, "vector_add");
  launch_add(a, b, out, variant, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_out", &add_out, "Checked vector addition into a separate output");
}
