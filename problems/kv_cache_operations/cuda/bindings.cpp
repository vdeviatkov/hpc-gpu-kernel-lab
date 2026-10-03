#include <torch/extension.h>

#include <cstdint>

void launch_kv_cache_append(const at::Tensor &k_new, const at::Tensor &v_new, at::Tensor k_cache,
                            at::Tensor v_cache, const at::Tensor &block_table,
                            const at::Tensor &positions);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kv_cache_append", &launch_kv_cache_append,
        "47 · KV-Cache Update and Paged Access (not implemented)");
}
