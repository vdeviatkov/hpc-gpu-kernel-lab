"""PyTorch reference and eager baseline for 10 · Gaussian Blur.

F.conv2d is already a cross-correlation (no kernel flip), and padding (KH//2, KW//2) with
odd kernel sizes gives the same-size, zero-padded output of the contract, for any kernel
size including one larger than the image. The reference always runs the direct 2-D
filter, never a separable factorization, so it is exact to the supplied kernel.

cuDNN may run FP32 convolutions in TF32 (10-bit mantissa) when
torch.backends.cudnn.allow_tf32 is set, which is PyTorch's default and is what the
source checker runs with. That would make the reference, not the backend under test, the
inaccurate side, so it is disabled here for this call only; the other cuDNN flags are kept.
"""

import torch
import torch.nn.functional as F


def gaussian_blur(image, kernel):
    (h, w), (kh, kw) = image.shape, kernel.shape
    cudnn = torch.backends.cudnn
    with cudnn.flags(
        enabled=cudnn.enabled,
        benchmark=cudnn.benchmark,
        deterministic=cudnn.deterministic,
        allow_tf32=False,
    ):
        out = F.conv2d(
            image.view(1, 1, h, w), kernel.view(1, 1, kh, kw), padding=(kh // 2, kw // 2)
        )
    # A fresh contiguous (1, 1, H, W) result; view keeps it fresh and owned.
    return out.view(h, w)
