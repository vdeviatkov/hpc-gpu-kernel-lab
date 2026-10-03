#include <torch/extension.h>

#include <cstdint>

void launch_causal_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                             at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("causal_attention", &launch_causal_attention,
        "41 · Causal Self-Attention (not implemented)");
}
