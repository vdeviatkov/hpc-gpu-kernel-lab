#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_reverse_(at::Tensor x);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reverse_", &launch_reverse_, "03 · Reverse Array (not implemented)");
}
