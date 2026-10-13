"""PyTorch reference and eager baselines for 04 · Interleave Arrays.

method="slices" is the statement's own reference and the oracle every other backend is
tested against: two strided copies into a fresh output. method="stack" is a second
baseline that builds the output in one contiguous-write kernel.
"""

import torch

def interleave(a, b, *, method="slices"):
    if method == "stack":
        return torch.stack((a, b), dim=1).reshape(-1)
    out = torch.empty(2 * a.numel(), dtype=a.dtype, device=a.device)
    out[0::2] = a
    out[1::2] = b
    return out
