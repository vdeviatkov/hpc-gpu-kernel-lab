#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_matmul(const at::Tensor &a, const at::Tensor &b, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &launch_matmul, "23 · Matrix Multiplication (not implemented)");
}
