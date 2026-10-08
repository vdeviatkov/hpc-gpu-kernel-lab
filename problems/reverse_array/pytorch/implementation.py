import torch


def reverse_(x):
    return x.copy_(torch.flip(x, [0]))
