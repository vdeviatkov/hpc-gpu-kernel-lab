"""08 · Matrix Transpose: public entry point.

Source contract verified 2026-10-24 at the LeetGPU link in README.md:
row-major fp32 input (rows, cols), row-major output (cols, rows),
out[j, i] = a[i, j]; 1 <= rows, cols <= 8192. Performance case: (7000, 6000).

The lab returns a fresh contiguous tensor on the input device, without mutation.
Lab extensions: fp16/bf16, empty dimensions, and dimensions above 8192.
Only contiguous strided 2-D input is accepted; storage offsets are allowed.
Transpose performs no arithmetic; all element bits must be preserved.
"""

import sys

import torch

from lab import dispatch

FUNCTION = "transpose"
VARIANTS = {
    # One thread per element: coalesced loads and strided stores, or the reverse.
    "cuda_naive_row": ("cuda", {"variant": 0}),
    "cuda_naive_col": ("cuda", {"variant": 1}),
    # A 32x32 shared tile so both sides coalesce: plain (bank conflicts on column reads),
    # padded to 32x33, or XOR-swizzled with no extra shared memory.
    "cuda_smem": ("cuda", {"variant": 2}),
    "cuda_smem_pad": ("cuda", {"variant": 3}),
    "cuda_smem_swizzle": ("cuda", {"variant": 4}),
    # A swizzled 16x256 tile moved with 16-byte packs, several per thread. Edge tiles and
    # inputs whose rows are not 16-byte aligned use scalar accesses.
    "cuda_vec_ilp": ("cuda", {"variant": 5}),
}
BACKENDS = ("pytorch", *VARIANTS, "cuda", "triton")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# (rtol, atol) against the PyTorch reference, per input dtype.
TOLERANCES = {
    torch.float32: (0, 0),
    torch.float16: (0, 0),
    torch.bfloat16: (0, 0),
}


def transpose(a, *, backend="pytorch", **options):
    """See the module docstring. `options` are backend launch parameters."""
    if not isinstance(a, torch.Tensor):
        raise TypeError("expected torch.Tensor")
    if a.layout != torch.strided or a.ndim != 2 or not a.is_contiguous():
        raise ValueError("expected a contiguous two-dimensional tensor")
    if a.dtype not in DTYPES.values():
        raise ValueError("supported dtypes: fp32, fp16, bf16")
    return dispatch.call(sys.modules[__name__], FUNCTION, backend, a, **options)
