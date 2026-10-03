#include <torch/extension.h>

#include <cstdint>

void launch_spmv(const at::Tensor &row_ptr, const at::Tensor &col_idx, const at::Tensor &values,
                 const at::Tensor &x, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// (see problems/vector_add/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmv", &launch_spmv, "22 · Sparse Matrix-Vector Multiplication (not implemented)");
}
