#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_gaussian_blur(const at::Tensor &image, const at::Tensor &kernel, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gaussian_blur", &launch_gaussian_blur, "10 · Gaussian Blur (not implemented)");
}
