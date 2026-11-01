#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_transpose(const at::Tensor &a, at::Tensor out);

void transpose(const at::Tensor &a, at::Tensor out) {
  for (const auto &t : {a, out}) {
    lab::check_cuda_matrix(t, "transpose");
    lab::check_plain(t, "transpose");
  }
  lab::check_float_dtype(a, "transpose");
  TORCH_CHECK(out.size(0) == a.size(1) && out.size(1) == a.size(0) &&
                  out.scalar_type() == a.scalar_type() && out.device() == a.device(),
              "transpose: out must be the reversed shape with the same dtype and device");
  lab::check_no_overlap(out, a, "transpose");
  launch_transpose(a, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transpose", &transpose, "08 · Matrix Transpose", pybind11::arg("a"),
        pybind11::arg("out"));
}
