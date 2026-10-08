#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_reduce_sum(const at::Tensor &x, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce_sum", &launch_reduce_sum, "13 · Reduction (not implemented)");
}
