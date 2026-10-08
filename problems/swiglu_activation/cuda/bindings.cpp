#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_swiglu(const at::Tensor &x, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swiglu", &launch_swiglu, "31 · Swish-Gated Linear Unit (not implemented)");
}
