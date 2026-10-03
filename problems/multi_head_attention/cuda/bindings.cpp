#include <torch/extension.h>

#include <cstdint>

void launch_mha(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mha", &launch_mha, "42 · Multi-Head Attention (not implemented)");
}
