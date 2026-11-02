"""08 · Matrix Transpose: every backend against the PyTorch reference."""

import pytest

from lab.data import Sampler
from lab.testing import (
    call_or_skip,
    case_id,
    check_against_reference,
    device,
    skip_if_unavailable,
    skip_unsupported_dtype,
)
from problems.matrix_transpose import api, cases


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
@pytest.mark.parametrize("backend", [b for b in api.BACKENDS if b != "pytorch"])
def test_matches_reference(backend, case, dtype):
    check_against_reference(api, cases, backend, case, dtype)


# Bit patterns a value-level comparison cannot tell apart, or that a kernel moving data
# through arithmetic would alter: signed zeros, infinities, quiet and signalling NaNs with
# distinct payloads and signs, the smallest and largest subnormals, the largest finite
# value and 1.0. The same set as 07 · Matrix Copy.
SPECIAL_BITS = {
    "fp32": [
        0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7FC00001,
        0xFFC12345, 0x7F800001, 0x00000001, 0x007FFFFF, 0x7F7FFFFF, 0x3F800000,
    ],
    "fp16": [
        0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x7E01,
        0xFD23, 0x7C01, 0x0001, 0x03FF, 0x7BFF, 0x3C00,
    ],
    "bf16": [
        0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7FC1,
        0xFFA5, 0x7F81, 0x0001, 0x007F, 0x7F7F, 0x3F80,
    ],
}  # fmt: skip


def bits(t):
    import torch

    return t.view(torch.int32 if t.element_size() == 4 else torch.int16)


def run(backend, a):
    skip_if_unavailable(backend, api)
    skip_unsupported_dtype(a.dtype)
    return call_or_skip(api, backend, (a.to(device()),))


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_special_values_bit_exact(backend, dtype):
    """Special patterns at every position of an odd-shaped matrix, beside random bits."""
    import torch

    name, dtype = dtype, api.DTYPES[dtype]
    integer = torch.int32 if dtype == torch.float32 else torch.int16
    width = 8 * torch.empty((), dtype=dtype).element_size()
    special = torch.tensor(
        [p - (1 << width) if p >> (width - 1) else p for p in SPECIAL_BITS[name]], dtype=integer
    )
    m, n = 7, 13
    info = torch.iinfo(integer)
    generator = torch.Generator().manual_seed(0)
    flat = torch.randint(info.min, info.max + 1, (m * n,), generator=generator).to(integer)
    # Rotate the specials through every row and column, including the tails.
    for start in range(0, m * n, len(special) + 1):
        stop = min(start + len(special), m * n)
        flat[start:stop] = special[: stop - start]
    a = flat.view(dtype).reshape(m, n)
    out = run(backend, a.clone())
    assert out.shape == (n, m) and out.dtype == dtype
    assert torch.equal(bits(out.cpu()), bits(a).t())


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_misaligned_input(backend, dtype):
    """A matrix starting one element into its storage is not 16-byte aligned: vector paths
    must fall back. 40 x 520 would otherwise take cuda_vec_ilp's pack path."""
    import torch

    m, n = 40, 520
    full = Sampler(api.DTYPES[dtype], device()).randn(1 + m * n)
    a = full[1:].view(m, n)
    assert a.is_contiguous() and a.data_ptr() % 16 != 0
    assert torch.equal(bits(run(backend, a).cpu()), bits(a.cpu()).t())


@pytest.mark.parametrize("tile", [(16, 128), (128, 16), (32, 32), (1, 1024)], ids=str)
def test_triton_tile_shapes(tile):
    import torch

    skip_if_unavailable("triton", api)
    a = Sampler(torch.float16, device()).randn(65, 129)
    block_m, block_n = tile
    out = api.transpose(a, backend="triton", block_m=block_m, block_n=block_n)
    assert torch.equal(bits(out), bits(a).t())


def test_triton_rejects_invalid_options():
    import torch

    skip_if_unavailable("triton", api)
    a = torch.zeros(4, 8, device="cuda")
    for options in [{"block_m": 0}, {"block_n": 48}, {"num_warps": 3}]:
        with pytest.raises(ValueError):
            api.transpose(a, backend="triton", **options)


def test_rejects_unknown_backend():
    dtype = next(iter(api.DTYPES.values()))
    args = cases.make_inputs(cases.TEST_CASES[0], Sampler(dtype, "cpu"))
    with pytest.raises(ValueError, match="unknown backend"):
        api.transpose(*args, backend="invalid")


@pytest.mark.parametrize("dtype", api.DTYPES.values())
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
def test_reference_contract(case, dtype):
    import torch

    m, n = case["m"], case["n"]
    # Offset storage is contiguous and must be accepted.
    a = torch.arange(m * n + 1, dtype=torch.float32)[1:].to(dtype).reshape(m, n)
    before = a.clone()
    out = api.transpose(a)
    expected = torch.empty((n, m), dtype=dtype)
    for i in range(m):
        for j in range(n):
            expected[j, i] = a[i, j]
    assert torch.equal(out, expected)
    assert out.shape == (n, m) and out.dtype == dtype and out.device == a.device
    assert out.is_contiguous()
    if a.numel():
        assert out.data_ptr() != a.data_ptr()
    out.fill_(-1)
    assert torch.equal(a, before)


@pytest.mark.parametrize("dtype", api.DTYPES.values())
def test_reference_preserves_bits(dtype):
    import torch

    integer = torch.int32 if dtype == torch.float32 else torch.int16
    # Exhaust every 16-bit pattern; sample random fp32 patterns plus signed zero,
    # infinities and distinct NaN payloads.
    if integer == torch.int16:
        raw = torch.arange(-32768, 32768, dtype=torch.int32).to(integer).reshape(256, 256)
    else:
        raw = torch.randint(-(2**31), 2**31, (32, 33), generator=torch.Generator().manual_seed(0))
        raw = raw.to(integer)
        raw[0, :6] = torch.tensor(
            [0, -(2**31), 0x7F800000, -8388608, 0x7FC00001, 0x7F800001], dtype=integer
        )
    out = api.transpose(raw.view(dtype)).view(integer)
    assert torch.equal(out, raw.t())


def test_rejects_invalid_inputs():
    import torch

    a = torch.zeros(3, 4)
    for bad in [a.flatten(), a[None], a.t(), a[:, ::2], a.double(), a.int(), a.to_sparse()]:
        with pytest.raises(ValueError):
            api.transpose(bad)
    with pytest.raises(TypeError):
        api.transpose([[1.0]])


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_gpu_backends_reject_cpu(backend):
    import torch

    with pytest.raises(ValueError, match="requires NVIDIA CUDA"):
        api.transpose(torch.zeros(2, 3), backend=backend)


def test_traffic_model():
    import torch

    c = {"m": 7000, "n": 6000}
    assert cases.logical_bytes(c, torch.float32) == 336_000_000
    assert cases.logical_bytes(c, torch.float16) == 168_000_000
    assert cases.flops(c) is None


CUDA_VARIANTS = {
    name: options["variant"] for name, (module, options) in api.VARIANTS.items() if module == "cuda"
}


def cuda_extension():
    skip_if_unavailable("cuda")
    from problems.matrix_transpose.cuda.implementation import extension

    return extension()


@pytest.mark.parametrize("offsets", [(1, 0), (0, 1), (1, 1)], ids=["input", "output", "both"])
@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("backend", CUDA_VARIANTS)
def test_binding_misaligned_buffers(backend, dtype, offsets):
    """Rows that are not 16-byte aligned must take cuda_vec_ilp's scalar path. The api
    always allocates an aligned output, so a misaligned output needs a direct call.
    Elements outside the output view must stay untouched."""
    import torch

    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    ext = cuda_extension()
    m, n = 40, 520
    a_offset, out_offset = offsets
    a = Sampler(dtype, "cuda").randn(1 + m * n)[a_offset : a_offset + m * n].view(m, n)
    out_buffer = torch.full((1 + m * n,), -7.0, dtype=dtype, device="cuda")
    out = out_buffer[out_offset : out_offset + m * n].view(n, m)
    assert (a.data_ptr() % 16 != 0) == bool(a_offset)
    assert (out.data_ptr() % 16 != 0) == bool(out_offset)
    ext.transpose(a, out, CUDA_VARIANTS[backend])
    assert torch.equal(
        out.view(torch.int16 if dtype != torch.float32 else torch.int32),
        a.t().view(torch.int16 if dtype != torch.float32 else torch.int32),
    )
    guard = out_buffer[0] if out_offset else out_buffer[-1]
    assert guard.item() == -7.0


def test_binding_rejects_invalid_arguments():
    import torch

    ext = cuda_extension()
    a = torch.zeros(4, 8, device="cuda")
    out = torch.empty(8, 4, device="cuda")
    shared = torch.zeros(40, device="cuda")
    bad_calls = [
        (a, out, -1),  # variant
        (a, out, 6),
        (a.flatten(), out.flatten(), 3),  # 1-D
        (a.t(), torch.empty(4, 8, device="cuda"), 3),  # non-contiguous input
        (a, torch.empty(4, 8, device="cuda"), 3),  # shape not reversed
        (a, out.half(), 3),  # dtype mismatch
        (a.double(), out.double(), 3),  # fp64
        (a.cpu(), out.cpu(), 3),  # host tensors
        (shared[:32].view(4, 8), shared[8:].view(8, 4), 3),  # partial overlap
    ]
    for args in bad_calls:
        with pytest.raises(RuntimeError):
            ext.transpose(*args)
