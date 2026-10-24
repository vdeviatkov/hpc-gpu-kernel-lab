"""PyTorch reference and eager baseline for 07 · Matrix Copy.

The source's own reference is `B[:] = A`. Here the output is allocated, so the plainest
equivalent is `clone`. For a contiguous CUDA tensor of one dtype, ATen's copy reduces to a
single cudaMemcpyAsync device-to-device, so this baseline is the driver's copy engine path
rather than a generated kernel: the same operation as lab.bench's `--copy-reference` row,
and the practical ceiling the custom kernels are measured against.
"""

import torch


def matrix_copy(a, *, method="reference"):
    if method != "reference":
        raise ValueError(f"unknown matrix_copy method: {method}")
    return a.clone(memory_format=torch.contiguous_format)
