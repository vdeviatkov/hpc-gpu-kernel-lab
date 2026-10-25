#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_matrix_copy(const at::Tensor &a, at::Tensor out, int64_t variant, int64_t threads);

void matrix_copy(const at::Tensor &a, at::Tensor out, int64_t variant, int64_t threads) {
  TORCH_CHECK(variant >= 0 && variant <= 7, "matrix_copy: invalid variant");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512,
              "matrix_copy: invalid block size");
  for (const auto &t : {a, out}) {
    lab::check_cuda_matrix(t, "matrix_copy");
    lab::check_plain(t, "matrix_copy");
  }
  lab::check_float_dtype(a, "matrix_copy");
  lab::check_same(out, a, "matrix_copy");
  lab::check_no_overlap(out, a, "matrix_copy");
  launch_matrix_copy(a, out, variant, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matrix_copy", &matrix_copy, "Row-major matrix copy", pybind11::arg("a"),
        pybind11::arg("out"), pybind11::arg("variant") = 2, pybind11::arg("threads") = 256);
}
