#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_swiglu_mlp(const at::Tensor &x, const at::Tensor &w_gate, const at::Tensor &w_up,
                       const at::Tensor &w_down, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swiglu_mlp", &launch_swiglu_mlp, "39 · SwiGLU MLP Block (not implemented)");
}
