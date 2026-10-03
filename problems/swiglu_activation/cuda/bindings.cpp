#include <torch/extension.h>

#include <cstdint>

void launch_swiglu(const at::Tensor &x, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swiglu", &launch_swiglu, "31 · Swish-Gated Linear Unit (not implemented)");
}
