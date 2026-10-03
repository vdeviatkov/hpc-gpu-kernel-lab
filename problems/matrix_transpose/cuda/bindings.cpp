#include <torch/extension.h>

#include <cstdint>

void launch_transpose(const at::Tensor &a, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transpose", &launch_transpose, "08 · Matrix Transpose (not implemented)");
}
