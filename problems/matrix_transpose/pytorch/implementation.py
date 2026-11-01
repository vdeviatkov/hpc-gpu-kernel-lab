"""Materialized PyTorch transpose: correctness reference and eager baseline."""

import torch

def transpose(a):
    # clone is mandatory: contiguous() alone aliases for singleton/empty dimensions.
    return a.t().clone(memory_format=torch.contiguous_format)

