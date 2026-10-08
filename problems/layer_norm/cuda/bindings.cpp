#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_layer_norm(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &bias,
                       double eps, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layer_norm", &launch_layer_norm, "34 · Layer Normalization (not implemented)");
}
