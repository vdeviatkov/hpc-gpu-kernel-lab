#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_matrix_copy(const at::Tensor &a, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matrix_copy", &launch_matrix_copy, "07 · Matrix Copy (not implemented)");
}
