#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_inclusive_scan(const at::Tensor &x, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inclusive_scan", &launch_inclusive_scan, "17 · Prefix Sum (not implemented)");
}
