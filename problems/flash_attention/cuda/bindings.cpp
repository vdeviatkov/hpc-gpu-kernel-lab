#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_flash_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                            bool causal, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_attention", &launch_flash_attention,
        "48 · FlashAttention-Style Online Attention (not implemented)");
}
