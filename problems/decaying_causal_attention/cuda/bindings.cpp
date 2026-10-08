#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_decaying_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                               double gamma, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decaying_attention", &launch_decaying_attention,
        "45 · Decaying Causal Attention (not implemented)");
}
