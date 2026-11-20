#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_gaussian_blur(const at::Tensor &image, const at::Tensor &kernel, at::Tensor out,
                          int64_t variant);
int64_t max_const_taps();

void gaussian_blur(const at::Tensor &image, const at::Tensor &kernel, at::Tensor out,
                   int64_t variant) {
  TORCH_CHECK(variant == 0 || variant == 1, "gaussian_blur: invalid variant");
  for (const auto &t : {image, kernel, out}) {
    lab::check_cuda_matrix(t, "gaussian_blur");
    lab::check_plain(t, "gaussian_blur");
  }
  lab::check_float_dtype(image, "gaussian_blur");
  lab::check_same(out, image, "gaussian_blur");
  TORCH_CHECK(kernel.scalar_type() == image.scalar_type() && kernel.device() == image.device(),
              "gaussian_blur: image and kernel must share dtype and device");
  TORCH_CHECK(image.numel() >= 1, "gaussian_blur: expected H, W >= 1");
  const int64_t kh = kernel.size(0), kw = kernel.size(1);
  TORCH_CHECK(kh >= 1 && kw >= 1 && kh % 2 == 1 && kw % 2 == 1,
              "gaussian_blur: expected odd kernel dimensions KH, KW >= 1");
  lab::check_no_overlap(out, image, "gaussian_blur");
  lab::check_no_overlap(out, kernel, "gaussian_blur");
  launch_gaussian_blur(image, kernel, out, variant);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gaussian_blur", &gaussian_blur, "10 · Gaussian Blur", pybind11::arg("image"),
        pybind11::arg("kernel"), pybind11::arg("out"), pybind11::arg("variant") = 0);
  m.def("max_const_taps", &max_const_taps, "cuda_direct_const kernel capacity in taps");
}
