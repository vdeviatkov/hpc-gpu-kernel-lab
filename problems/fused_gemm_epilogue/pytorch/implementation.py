"""PyTorch reference and eager baseline for 29 · Fused GEMM + Bias + Activation.

Not implemented yet. Every other backend is tested against this function, so
write the plainest correct version of the contract in api.py.
"""


def linear_relu(x, w, bias):
    raise NotImplementedError(
        "linear_relu: PyTorch reference not implemented (pytorch/implementation.py)"
    )
