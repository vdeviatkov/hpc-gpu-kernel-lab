"""Checked eager API; optional compilers are imported only when selected."""

from importlib import import_module

import torch

# CUDA variants form a 2x2 design: {scalar, 16-byte vector} x {direct, grid-stride}.
CUDA_VARIANTS = {"cuda_scalar": 0, "cuda_grid_stride": 1, "cuda_vec_grid_stride": 2, "cuda_vec": 3}
BACKENDS = ("pytorch", *CUDA_VARIANTS, "triton")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
TOLERANCES = {
    torch.float32: (1e-6, 1e-7),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (8e-3, 8e-3),
}


def validate(a, b, out=None, *, gpu=False):
    for x in (a, b) if out is None else (a, b, out):
        if not isinstance(x, torch.Tensor):
            raise TypeError("expected torch.Tensor")
        if x.layout != torch.strided or x.ndim != 1 or not x.is_contiguous():
            raise ValueError("expected contiguous one-dimensional tensors")
        if x.dtype not in DTYPES.values():
            raise ValueError("supported dtypes: fp32, fp16, bf16")
        if x.shape != a.shape or x.dtype != a.dtype or x.device != a.device:
            raise ValueError("shape, dtype, and device must match")
        if x.requires_grad:
            raise ValueError("this experiment is forward-only; detach inputs first")
        if x.is_neg() or x.is_conj():
            raise ValueError("unresolved negative/conjugate views are unsupported")
    if gpu and (a.device.type != "cuda" or torch.version.cuda is None):
        raise ValueError("backend requires NVIDIA CUDA tensors")
    if out is not None and out.numel():
        lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        for x in (a, b):
            if lo < x.data_ptr() + x.numel() * x.element_size() and x.data_ptr() < hi:
                raise ValueError("output must not overlap either input")


def add(a, b, *, backend="pytorch", out=None, threads=256, block_size=1024, num_warps=4):
    """Add equal 1-D vectors, allocating output unless supplied. No broadcasting/autograd."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    validate(a, b, out, gpu=backend != "pytorch")
    if backend == "pytorch":
        return import_module(f"{__package__}.pytorch.implementation").add(a, b, out=out)
    if backend in CUDA_VARIANTS:
        if threads not in (128, 256, 512):
            raise ValueError("threads must be 128, 256, or 512")
        return import_module(f"{__package__}.cuda.implementation").add(
            a, b, out=out, variant=CUDA_VARIANTS[backend], threads=threads
        )
    if block_size not in (256, 512, 1024, 2048) or num_warps not in (4, 8):
        raise ValueError("block_size must be 256/512/1024/2048; num_warps must be 4/8")
    return import_module(f"{__package__}.triton.implementation").add(
        a, b, out=out, block_size=block_size, num_warps=num_warps
    )
