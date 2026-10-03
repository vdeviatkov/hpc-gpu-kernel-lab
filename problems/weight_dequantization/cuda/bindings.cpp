#include <torch/extension.h>

#include <cstdint>

void launch_dequantize(const at::Tensor &q, const at::Tensor &scale, int64_t quant_block,
                       at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequantize", &launch_dequantize, "12 · Weight Dequantization (not implemented)");
}
