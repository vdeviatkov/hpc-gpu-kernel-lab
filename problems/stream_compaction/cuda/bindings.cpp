#include <torch/extension.h>

#include <cstdint>

void launch_compact_positive(const at::Tensor &x, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compact_positive", &launch_compact_positive, "19 · Stream Compaction (not implemented)");
}
