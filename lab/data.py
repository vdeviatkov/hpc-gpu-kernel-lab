"""Reproducible input sampling shared by tests and benchmarks.

Values are drawn on the CPU from a seeded generator, then moved to the target
device, so every backend and device sees identical inputs.
"""

import torch


class Sampler:
    def __init__(self, dtype, device, seed=0):
        self.dtype = dtype
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def _place(self, tensor, dtype):
        return tensor.to(dtype or self.dtype).to(self.device)

    def randn(self, *shape, dtype=None, scale=1.0):
        return self._place(torch.randn(shape, generator=self.generator) * scale, dtype)

    def uniform(self, low, high, *shape, dtype=None):
        values = torch.rand(shape, generator=self.generator) * (high - low) + low
        return self._place(values, dtype)

    def randint(self, low, high, *shape, dtype=torch.int32):
        values = torch.randint(low, high, shape, generator=self.generator, dtype=torch.int64)
        return values.to(dtype).to(self.device)

    def sorted(self, *shape, dtype=None):
        return self._place(torch.randn(shape, generator=self.generator).sort().values, dtype)

    def csr(self, rows, cols, nnz_per_row, dtype=None):
        """Random CSR matrix as (row_ptr int32, col_idx int32, values).

        Each row draws one column per stratum from 2 * nnz_per_row strata and keeps
        each with probability 1/2: rows average nnz_per_row entries with varying
        lengths, and columns are distinct and sorted. Memory is O(rows * nnz_per_row).
        """
        strata = max(1, min(cols, 2 * nnz_per_row))
        width = cols // strata
        offsets = torch.randint(0, width, (rows, strata), generator=self.generator)
        candidates = offsets + torch.arange(strata) * width
        keep = torch.rand(rows, strata, generator=self.generator) < 0.5
        col_idx = candidates[keep]
        row_ptr = torch.zeros(rows + 1, dtype=torch.int64)
        row_ptr[1:] = keep.sum(1).cumsum(0)
        values = torch.randn(col_idx.numel(), generator=self.generator)
        return (
            row_ptr.to(torch.int32).to(self.device),
            col_idx.to(torch.int32).to(self.device),
            self._place(values, dtype),
        )

    def permutation(self, n, dtype=torch.int32):
        return torch.randperm(n, generator=self.generator).to(dtype).to(self.device)


def clone_tensors(args):
    """Fresh copies of tensor arguments, so in-place backends cannot affect each other."""
    return tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)


def element_size(dtype):
    return torch.empty((), dtype=dtype).element_size()
