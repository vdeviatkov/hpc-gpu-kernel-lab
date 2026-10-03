#include <torch/extension.h>

#include <cstdint>

void launch_route_tokens(const at::Tensor &x, const at::Tensor &router_logits, int64_t top_k,
                         at::Tensor dispatched, at::Tensor expert_offsets, at::Tensor token_index,
                         at::Tensor weights);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route_tokens", &launch_route_tokens,
        "50 · MoE Token Routing and Dispatch (not implemented)");
}
