#include <torch/extension.h>

#include <cstdint>

void launch_rgb_to_grayscale(const at::Tensor &image, at::Tensor out, int64_t variant);

// Repeats the Python api's checks so a direct call into the extension cannot launch on bad
// input: an (H, W, 3) contiguous CUDA image and an (H, W) contiguous output of the same
// dtype and device.
void rgb_to_grayscale_out(const at::Tensor &image, at::Tensor out, int64_t variant) {
  TORCH_CHECK(variant >= 0 && variant <= 3, "rgb_to_grayscale: invalid variant");
  TORCH_CHECK(image.is_cuda() && image.dim() == 3 && image.size(2) == 3 && image.is_contiguous(),
              "rgb_to_grayscale: expected a contiguous (H, W, 3) CUDA image");
  const auto type = image.scalar_type();
  TORCH_CHECK(type == at::kFloat || type == at::kHalf || type == at::kBFloat16,
              "rgb_to_grayscale: supported dtypes are fp32, fp16 and bf16");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2 &&
                  out.size(0) == image.size(0) && out.size(1) == image.size(1) &&
                  out.scalar_type() == type && out.device() == image.device(),
              "rgb_to_grayscale: output must be a contiguous (H, W) tensor like the image");
  TORCH_CHECK(!image.requires_grad(), "rgb_to_grayscale: autograd is unsupported");
  launch_rgb_to_grayscale(image, out, variant);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rgb_to_grayscale_out", &rgb_to_grayscale_out,
        "05 · RGB to Grayscale into a separate output");
}
