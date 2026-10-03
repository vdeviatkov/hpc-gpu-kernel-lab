#include <torch/extension.h>

#include <cstdint>

void launch_gemm(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c, double alpha,
                 double beta, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &launch_gemm, "25 · General Matrix Multiplication (GEMM) — FP16 (not implemented)");
}
