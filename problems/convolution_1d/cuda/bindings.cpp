#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_conv1d(const at::Tensor &x, const at::Tensor &w, at::Tensor out, int64_t variant,
                   int64_t reg_outputs);
int64_t max_const_taps();

void conv1d(const at::Tensor &x, const at::Tensor &w, at::Tensor out, int64_t variant,
            int64_t reg_outputs) {
  TORCH_CHECK(variant >= 0 && variant <= 5, "conv1d: invalid variant");
  TORCH_CHECK(reg_outputs == 2 || reg_outputs == 4 || reg_outputs == 8 || reg_outputs == 16,
              "conv1d: reg_outputs must be 2, 4, 8 or 16");
  for (const auto &t : {x, w, out}) {
    lab::check_cuda_vector(t, "conv1d");
    lab::check_plain(t, "conv1d");
  }
  lab::check_float_dtype(x, "conv1d");
  TORCH_CHECK(w.scalar_type() == x.scalar_type() && out.scalar_type() == x.scalar_type() &&
                  w.device() == x.device() && out.device() == x.device(),
              "conv1d: x, w and out must share dtype and device");
  TORCH_CHECK(w.numel() >= 1 && w.numel() <= x.numel(), "conv1d: expected 1 <= K <= N");
  TORCH_CHECK(out.numel() == x.numel() - w.numel() + 1, "conv1d: out must have N - K + 1 elements");
  lab::check_no_overlap(out, x, "conv1d");
  lab::check_no_overlap(out, w, "conv1d");
  launch_conv1d(x, w, out, variant, reg_outputs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv1d", &conv1d, "09 · 1D Convolution", pybind11::arg("x"), pybind11::arg("w"),
        pybind11::arg("out"), pybind11::arg("variant") = 3,
        pybind11::arg("reg_outputs") = 8);
  m.def("max_const_taps", &max_const_taps, "cuda_const filter capacity in taps");
}
