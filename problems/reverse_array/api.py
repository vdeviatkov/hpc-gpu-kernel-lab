"""03 · Reverse Array: public entry point.

Contract, verified against the LeetGPU statement (challenges/easy/19_reverse_array):

    x[i] <-> x[N - 1 - i]   for 0 <= i < N / 2, in place

- Source: contiguous 1-D FP32, 1 <= N <= 100,000,000, result stored back into the
  input. The performance test uses N = 25,000,000 with inputs uniform in
  [-1000, 1000]; the reference is `input[:] = torch.flip(input, [0])`.
- For odd N the middle element stays where it is.
- Lab extensions: FP16/BF16 and N = 0.
- The function returns the same tensor it was given, now reversed.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "reverse_"
VARIANTS = {
    "cuda_pair": ("cuda", {"variant": 0}),  # one thread per element pair
    # One thread per pair of 16-byte packs; needs N % (16 / element size) == 0 and an
    # aligned pointer, otherwise it runs cuda_pair.
    "cuda_vec": ("cuda", {"variant": 1}),
    # One block per tile of pairs, staged through shared memory: 16-byte accesses for any
    # N (needs only an aligned pointer, otherwise it runs cuda_pair).
    "cuda_tile": ("cuda", {"variant": 2}),
    # cuda_tile with asynchronous global -> shared copies (cp.async) for interior packs.
    "cuda_tile_async": ("cuda", {"variant": 3}),
    # Triton: each element owns a pair, back half addressed in descending order.
    "triton_pair": ("triton", {"flip": False}),
    # Triton: both halves loaded in ascending order, reversed in registers with tl.flip.
    "triton_flip": ("triton", {"flip": True}),
}
BACKENDS = ("pytorch", *VARIANTS)
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference. Reversal only moves values, so every
# backend must match bit for bit.
TOLERANCES = {
    torch.float32: (0, 0),
    torch.float16: (0, 0),
    torch.bfloat16: (0, 0),
}

# The function updates its input in place.
MUTATES = True


def reverse_(x, *, backend="pytorch", **options):
    """Reverse `x` in place and return it. `options` are backend launch parameters."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected torch.Tensor")
    if x.layout != torch.strided or x.ndim != 1 or not x.is_contiguous():
        raise ValueError("expected a contiguous one-dimensional tensor")
    if x.dtype not in DTYPES.values():
        raise ValueError("supported dtypes: fp32, fp16, bf16")
    if x.requires_grad:
        raise ValueError("in-place reversal does not support autograd; detach first")
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, x, **options)
