"""PyTorch reference and eager baseline for 36 · Fused Residual Add and RMS Norm.

Not implemented yet. Every other backend is tested against this function, so
write the plainest correct version of the contract in api.py.
"""


def fused_add_rms_norm(x, residual, weight, eps):
    raise NotImplementedError(
        "fused_add_rms_norm: PyTorch reference not implemented (pytorch/implementation.py)"
    )
