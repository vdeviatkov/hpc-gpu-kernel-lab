#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_merge(const at::Tensor &a, const at::Tensor &b, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge", &launch_merge, "20 · Parallel Merge (not implemented)");
}
