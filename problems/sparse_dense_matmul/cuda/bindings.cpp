#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_spmm(const at::Tensor &row_ptr, const at::Tensor &col_idx, const at::Tensor &values,
                 const at::Tensor &b, at::Tensor out);

// TODO: validate shapes, dtypes, devices and contiguity at the C++ boundary
// with the helpers in lab/checks.h (see problems/relu/cuda/bindings.cpp).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm", &launch_spmm, "28 · Sparse Matrix-Dense Matrix Multiplication (not implemented)");
}
