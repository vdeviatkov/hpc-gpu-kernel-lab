#include <torch/extension.h>

#include <cstdint>

void launch_mse(const at::Tensor &pred, const at::Tensor &target, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mse", &launch_mse, "15 · Mean Squared Error (not implemented)");
}
