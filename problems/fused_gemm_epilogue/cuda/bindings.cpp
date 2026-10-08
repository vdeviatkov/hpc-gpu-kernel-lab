#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_linear_relu(const at::Tensor &x, const at::Tensor &w, const at::Tensor &bias,
                        at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_relu", &launch_linear_relu,
        "29 · Fused GEMM + Bias + Activation (not implemented)");
}
