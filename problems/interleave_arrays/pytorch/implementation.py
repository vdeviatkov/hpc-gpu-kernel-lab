"""PyTorch reference and eager baseline for 04 · Interleave Arrays.

The statement's own reference: two strided copies into a fresh output. Every other
backend is tested against this function.
"""

import torch


def interleave(a, b):
    out = torch.empty(2 * a.numel(), dtype=a.dtype, device=a.device)
    out[0::2] = a
    out[1::2] = b
    return out
