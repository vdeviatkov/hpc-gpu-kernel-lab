#include <torch/extension.h>

#include <cstdint>

void launch_fused_add_rms_norm(const at::Tensor &x, const at::Tensor &residual,
                               const at::Tensor &weight, double eps, at::Tensor out,
                               at::Tensor residual_out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_add_rms_norm", &launch_fused_add_rms_norm,
        "36 · Fused Residual Add and RMS Norm (not implemented)");
}
