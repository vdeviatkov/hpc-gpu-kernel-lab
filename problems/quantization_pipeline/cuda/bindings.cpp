#include <torch/extension.h>

#include <cstdint>

void launch_quantize(const at::Tensor &x, int64_t group_size, at::Tensor q, at::Tensor scale);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &launch_quantize, "46 · Quantize / Dequantize Pipeline (not implemented)");
}
