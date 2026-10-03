#include <torch/extension.h>

#include <cstdint>

void launch_alibi_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                            const at::Tensor &slopes, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("alibi_attention", &launch_alibi_attention,
        "43 · Attention with Linear Biases (not implemented)");
}
