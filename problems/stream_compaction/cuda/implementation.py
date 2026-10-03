"""CUDA backend for 19 · Stream Compaction. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def compact_positive(x):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "compact_positive: CUDA backend not implemented (cuda/kernels.cu)"
        )
    # TODO: Allocate the worst case, count kept elements on device, return out[:count].
    out = torch.empty_like(x)
    load_extension("gpu_lab_stream_compaction", Path(__file__).parent).compact_positive(x, out)
    return out
