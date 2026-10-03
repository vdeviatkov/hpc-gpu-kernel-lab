#include <torch/extension.h>

#include <cstdint>

void launch_paged_attention(const at::Tensor &q, const at::Tensor &k_cache,
                            const at::Tensor &v_cache, const at::Tensor &block_table,
                            const at::Tensor &context_lens, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attention", &launch_paged_attention, "49 · Paged Attention (not implemented)");
}
