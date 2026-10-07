#include <torch/extension.h>

#include <cstdint>

void launch_relu(const at::Tensor &x, at::Tensor out, int64_t variant, int64_t threads);

void relu_out(const at::Tensor &x, at::Tensor out, int64_t variant, int64_t threads) {
  TORCH_CHECK(variant >= 0 && variant <= 3, "relu: invalid variant");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "relu: invalid block size");
  for (const auto &t : {x, out}) {
    TORCH_CHECK(t.is_cuda() && t.layout() == c10::kStrided && t.dim() == 1 && t.is_contiguous(),
                "relu: expected contiguous 1-D CUDA tensors");
    TORCH_CHECK(t.sizes() == x.sizes() && t.scalar_type() == x.scalar_type() &&
                    t.device() == x.device(),
                "relu: shape, dtype and device must match");
  }
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kBFloat16,
              "relu: supported dtypes: fp32, fp16, bf16");
  launch_relu(x, out, variant, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_out", &relu_out, "02 · ReLU into a separate output");
}
