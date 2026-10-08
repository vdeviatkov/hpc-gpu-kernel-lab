#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_relu(const at::Tensor &x, at::Tensor out, int64_t variant, int64_t threads);

void relu_out(const at::Tensor &x, at::Tensor out, int64_t variant, int64_t threads) {
  TORCH_CHECK(variant >= 0 && variant <= 3, "relu: invalid variant");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "relu: invalid block size");
  for (const auto &t : {x, out}) {
    lab::check_cuda_vector(t, "relu");
    lab::check_same(t, x, "relu");
  }
  lab::check_float_dtype(x, "relu");
  launch_relu(x, out, variant, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_out", &relu_out, "02 · ReLU into a separate output");
}
