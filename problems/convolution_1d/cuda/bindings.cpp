#include <torch/extension.h>

#include <cstdint>

void launch_conv1d(const at::Tensor &x, const at::Tensor &w, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv1d", &launch_conv1d, "09 · 1D Convolution (not implemented)");
}
