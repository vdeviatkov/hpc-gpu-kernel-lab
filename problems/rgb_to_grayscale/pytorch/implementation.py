"""PyTorch reference and eager baseline for 05 · RGB to Grayscale.

The statement's formula with its exact coefficients, computed in FP32 and rounded to the
image dtype. Every other backend is tested against this function.
"""


def rgb_to_grayscale(image):
    rgb = image.float()
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.to(image.dtype)
