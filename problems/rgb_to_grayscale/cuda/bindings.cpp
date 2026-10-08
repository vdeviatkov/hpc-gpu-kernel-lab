#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_rgb_to_grayscale(const at::Tensor &image, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rgb_to_grayscale", &launch_rgb_to_grayscale, "05 · RGB to Grayscale (not implemented)");
}
