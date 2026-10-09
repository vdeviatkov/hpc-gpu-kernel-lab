#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_reverse_(at::Tensor x, int64_t variant, int64_t threads);

void reverse_(at::Tensor x, int64_t variant, int64_t threads) {
  TORCH_CHECK(variant >= 0 && variant <= 3, "reverse_: invalid variant");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "reverse_: invalid block size");
  lab::check_cuda_vector(x, "reverse_");
  lab::check_float_dtype(x, "reverse_");
  lab::check_plain(x, "reverse_");
  launch_reverse_(x, variant, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reverse_", &reverse_, "03 · Reverse Array, in place");
}
