"""Native eager baseline, also usable on CPU for contract tests."""

import torch


def add(a, b, *, out=None):
    return torch.add(a, b, out=out)
