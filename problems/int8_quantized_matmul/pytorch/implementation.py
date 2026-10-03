"""PyTorch reference and eager baseline for 27 · INT8 Quantized MatMul.

Not implemented yet. Every other backend is tested against this function, so
write the plainest correct version of the contract in api.py.
"""


def int8_matmul(a, b, scale_a, scale_b, zero_a, zero_b):
    raise NotImplementedError(
        "int8_matmul: PyTorch reference not implemented (pytorch/implementation.py)"
    )
