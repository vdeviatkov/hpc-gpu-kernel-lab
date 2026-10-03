#include <torch/extension.h>

#include <cstdint>

void launch_histogram(const at::Tensor &x, int64_t bins, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("histogram", &launch_histogram, "18 · Histogramming (not implemented)");
}
