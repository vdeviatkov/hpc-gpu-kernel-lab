#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_cross_entropy(const at::Tensor &logits, const at::Tensor &labels, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cross_entropy", &launch_cross_entropy,
        "37 · Categorical Cross Entropy Loss (not implemented)");
}
