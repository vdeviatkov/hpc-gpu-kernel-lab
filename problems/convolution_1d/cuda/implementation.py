"""CUDA conv1d variants, built lazily and launched on the current PyTorch stream.

variant 0 (cuda_naive): x and w from global memory. variant 1 (cuda_const): w staged as
FP32 in __constant__ memory, at most MAX_CONST_TAPS taps. variant 2 (cuda_smem): a shared
tile of x per block, filter walked in chunks of SMEM_CHUNK taps. variant 3
(cuda_smem_reg, the default): the shared tile plus REG_OUTPUTS outputs per thread from a
sliding register window. variant 4 (cuda_unrolled): compile-time K in UNROLLED_K, fully
unrolled with w in registers. variant 5 (cuda_large_k): variant 3 with each filter chunk
also staged in shared memory. See cuda/kernels.cu.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = True
MAX_CONST_TAPS = 16384  # 64 KiB of FP32; must match kMaxConstTaps in kernels.cu
SMEM_CHUNK = 1024  # must match kChunk in kernels.cu
REG_OUTPUTS = 8  # default R; must match kRegOutputs in kernels.cu
REG_OUTPUT_CHOICES = (2, 4, 8, 16)  # compiled R for variants 3 and 5
UNROLLED_K = (3, 7, 31)  # must match the conv1d_unrolled instantiations in kernels.cu


def extension():
    return load_extension("gpu_lab_convolution_1d", Path(__file__).parent)


def conv1d(x, w, *, variant=3, reg_outputs=REG_OUTPUTS):
    """reg_outputs (R) applies to cuda_smem_reg and cuda_large_k only."""
    if variant not in range(6):
        raise ValueError(
            "variant must be 0 (naive), 1 (const), 2 (smem), 3 (smem_reg), 4 (unrolled) "
            "or 5 (large_k)"
        )
    if variant == 4 and w.numel() not in UNROLLED_K:
        # Not an invalid input, just a K with no compiled specialization: tests and lab.bench
        # report NotImplementedError as "not implemented" for this case.
        raise NotImplementedError(f"cuda_unrolled is specialized for K in {UNROLLED_K}")
    if reg_outputs not in REG_OUTPUT_CHOICES:
        raise ValueError(f"reg_outputs must be one of {REG_OUTPUT_CHOICES}")
    if reg_outputs != REG_OUTPUTS and variant not in (3, 5):
        raise ValueError("reg_outputs applies only to cuda_smem_reg and cuda_large_k")
    if variant == 1 and w.numel() > MAX_CONST_TAPS:
        raise ValueError(f"cuda_const holds at most {MAX_CONST_TAPS} filter taps")
    out = torch.empty(x.numel() - w.numel() + 1, dtype=x.dtype, device=x.device)
    extension().conv1d(x, w, out, variant, reg_outputs)
    return out
