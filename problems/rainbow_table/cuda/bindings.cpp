#include <torch/extension.h>

#include <cstdint>

void launch_rainbow_table(const at::Tensor &x, int64_t rounds, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rainbow_table", &launch_rainbow_table, "06 · Rainbow Table (not implemented)");
}
