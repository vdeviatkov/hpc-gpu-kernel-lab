#include <torch/extension.h>

#include <cstdint>

void launch_rope(const at::Tensor &x, const at::Tensor &cos, const at::Tensor &sin, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope", &launch_rope, "38 · Rotary Positional Embedding (not implemented)");
}
