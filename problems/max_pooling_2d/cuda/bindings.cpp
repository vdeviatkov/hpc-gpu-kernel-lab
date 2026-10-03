#include <torch/extension.h>

#include <cstdint>

void launch_max_pool2d(const at::Tensor &x, int64_t kernel_size, int64_t stride, int64_t padding,
                       at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_pool2d", &launch_max_pool2d, "11 · 2D Max Pooling (not implemented)");
}
