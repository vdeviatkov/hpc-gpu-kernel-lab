#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cstdint>

void launch_add(const at::Tensor &a, const at::Tensor &b, at::Tensor &out, int variant,
                int threads);

void add_out(const at::Tensor &a, const at::Tensor &b, at::Tensor out, int variant, int threads) {
  TORCH_CHECK(variant >= 0 && variant <= 3, "invalid variant");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "invalid block size");
  for (const auto &x : {a, b, out}) {
    TORCH_CHECK(x.is_cuda() && x.layout() == c10::kStrided && x.dim() == 1 && x.is_contiguous(),
                "expected contiguous 1-D CUDA tensors");
    TORCH_CHECK(x.sizes() == a.sizes() && x.scalar_type() == a.scalar_type() &&
                    x.device() == a.device(),
                "shape, dtype and device must match");
    TORCH_CHECK(!x.requires_grad() && !x.is_neg() && !x.is_conj(),
                "autograd and unresolved views are unsupported");
  }
  TORCH_CHECK(a.scalar_type() == at::kFloat || a.scalar_type() == at::kHalf ||
                  a.scalar_type() == at::kBFloat16,
              "supported dtypes: fp32, fp16, bf16");
  if (a.numel() == 0)
    return;
  const auto begin = reinterpret_cast<uintptr_t>(out.data_ptr());
  const auto end = begin + out.nbytes();
  for (const auto &x : {a, b}) {
    const auto p = reinterpret_cast<uintptr_t>(x.data_ptr());
    TORCH_CHECK(!(begin < p + x.nbytes() && p < end), "output overlaps input");
  }
  launch_add(a, b, out, variant, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_out", &add_out, "Checked vector addition into a separate output");
}
