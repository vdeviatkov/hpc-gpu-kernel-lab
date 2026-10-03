#include <torch/extension.h>

#include <cstdint>

void launch_matmul(const at::Tensor &a, const at::Tensor &b, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &launch_matmul, "23 · Matrix Multiplication (not implemented)");
}
