#include <torch/extension.h>

#include <cstdint>

void launch_gemv(const at::Tensor &a, const at::Tensor &x, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &launch_gemv, "21 · Dense GEMV (not implemented)");
}
