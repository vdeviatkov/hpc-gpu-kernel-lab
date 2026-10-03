#include <torch/extension.h>

#include <cstdint>

void launch_rms_norm(const at::Tensor &x, const at::Tensor &weight, double eps, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &launch_rms_norm, "33 · RMS Normalization (not implemented)");
}
