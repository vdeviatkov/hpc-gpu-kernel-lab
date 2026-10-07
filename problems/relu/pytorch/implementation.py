"""PyTorch reference and eager baseline for 02 · ReLU.

Every other backend is tested against this function.
"""

import torch


def relu(x):
    return torch.relu(x)
