"""04 · Interleave Arrays: public entry point.

Contract, verified against the LeetGPU statement (challenges/easy/63_interleave):

    out[2*i] = a[i],  out[2*i + 1] = b[i]   for 0 <= i < N   (out has 2N elements)

- Source: contiguous 1-D FP32 inputs of equal length, 1 <= N <= 50,000,000, written to a
  separate output. The performance test uses N = 25,000,000 with normally distributed
  inputs; the reference is `output[0::2] = A; output[1::2] = B`.
- Lab extensions: FP16/BF16 and N = 0.
- The function returns a new tensor; the inputs are never modified. a and b may be the
  same tensor.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "interleave"
BACKENDS = ("pytorch", "cuda", "triton")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference. Interleaving only moves values, so every
# backend must match bit for bit.
TOLERANCES = {
    torch.float32: (0, 0),
    torch.float16: (0, 0),
    torch.bfloat16: (0, 0),
}


def interleave(a, b, *, backend="pytorch", **options):
    """Return [a[0], b[0], a[1], b[1], ...]. `options` are backend launch parameters."""
    for x in (a, b):
        if not isinstance(x, torch.Tensor):
            raise TypeError("expected torch.Tensor inputs")
        if x.layout != torch.strided or x.ndim != 1 or not x.is_contiguous():
            raise ValueError("expected contiguous one-dimensional tensors")
        if x.dtype not in DTYPES.values():
            raise ValueError("supported dtypes: fp32, fp16, bf16")
        if x.requires_grad:
            raise ValueError("this operation is forward-only; detach inputs first")
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("a and b must have the same length, dtype and device")
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, a, b, **options)
