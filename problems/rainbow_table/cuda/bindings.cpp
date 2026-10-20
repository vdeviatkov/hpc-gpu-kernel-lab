#include <torch/extension.h>

#include <cstdint>

#include "lab/checks.h"

void launch_rainbow_table(const at::Tensor &x, int64_t rounds, at::Tensor out, int64_t variant,
                          int64_t threads);

void rainbow_table(const at::Tensor &x, int64_t rounds, at::Tensor out, int64_t variant,
                   int64_t threads) {
  TORCH_CHECK(rounds >= 0, "rainbow_table: rounds must be non-negative");
  TORCH_CHECK(variant >= 0 && variant <= 5, "rainbow_table: invalid variant");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512,
              "rainbow_table: invalid block size");
  for (const auto &t : {x, out}) {
    lab::check_cuda_vector(t, "rainbow_table");
    lab::check_plain(t, "rainbow_table");
  }
  TORCH_CHECK(x.scalar_type() == at::kInt && out.scalar_type() == at::kUInt32,
              "rainbow_table: expected int32 input and uint32 output");
  TORCH_CHECK(x.sizes() == out.sizes() && x.device() == out.device(),
              "rainbow_table: output shape and device must match input");
  lab::check_no_overlap(out, x, "rainbow_table");
  launch_rainbow_table(x, rounds, out, variant, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rainbow_table", &rainbow_table, "Iterated FNV-1a hash", pybind11::arg("x"),
        pybind11::arg("rounds"), pybind11::arg("out"), pybind11::arg("variant") = 1,
        pybind11::arg("threads") = 256);
}
