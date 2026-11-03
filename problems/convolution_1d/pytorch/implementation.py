"""PyTorch reference and eager baseline for 09 · 1D Convolution.

F.conv1d is already a cross-correlation (no filter flip), so it matches the contract
directly. cuDNN may run FP32 convolutions in TF32 (10-bit mantissa) when
torch.backends.cudnn.allow_tf32 is set, which is PyTorch's default. That would make the
reference, not the backend under test, the inaccurate side, so it is disabled here for
this call only; the other cuDNN flags are kept as they are.
"""

import torch
import torch.nn.functional as F


def conv1d(x, w):
    cudnn = torch.backends.cudnn
    with cudnn.flags(
        enabled=cudnn.enabled,
        benchmark=cudnn.benchmark,
        deterministic=cudnn.deterministic,
        allow_tf32=False,
    ):
        # A fresh contiguous (1, 1, N - K + 1) result; view(-1) keeps it fresh and owned.
        return F.conv1d(x.view(1, 1, -1), w.view(1, 1, -1)).view(-1)
