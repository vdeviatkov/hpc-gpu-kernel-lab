"""07 · Matrix Copy: source examples, bit-exact special values, every backend vs the
reference."""

import pytest
import torch

from lab.data import Sampler
from lab.testing import (
    call_or_skip,
    case_id,
    check_against_reference,
    device,
    skip_if_unavailable,
    skip_unsupported_dtype,
)
from problems.matrix_copy import api, cases

# The integer type each element dtype is reinterpreted as for bit comparisons.
BITS = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}

# Bit patterns a value-level comparison cannot tell apart, or that a kernel moving data
# through arithmetic would alter: signed zeros, infinities, quiet and signalling NaNs with
# distinct payloads and signs, the smallest and largest subnormals, the largest finite
# value and 1.0.
SPECIAL_BITS = {
    torch.float32: [
        0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7FC00001,
        0xFFC12345, 0x7F800001, 0x00000001, 0x007FFFFF, 0x7F7FFFFF, 0x3F800000,
    ],
    torch.float16: [
        0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x7E01,
        0xFD23, 0x7C01, 0x0001, 0x03FF, 0x7BFF, 0x3C00,
    ],
    torch.bfloat16: [
        0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7FC1,
        0xFFA5, 0x7F81, 0x0001, 0x007F, 0x7F7F, 0x3F80,
    ],
}  # fmt: skip


def bits(t):
    return t.view(BITS[t.dtype])


def from_bits(patterns, dtype):
    """A tensor of `dtype` holding the given unsigned bit patterns exactly."""
    width = 8 * torch.empty((), dtype=dtype).element_size()
    signed = [p - (1 << width) if p >> (width - 1) else p for p in patterns]
    return torch.tensor(signed, dtype=BITS[dtype]).view(dtype)


def assert_same_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(bits(actual.cpu()), bits(expected.cpu()))


def run(backend, a):
    skip_if_unavailable(backend, api)
    skip_unsupported_dtype(a.dtype)
    return call_or_skip(api, backend, (a.to(device()),))


# The statement's worked examples; its output equals its input.
SOURCE_EXAMPLES = {
    "example1": [[1.0, 2.0], [3.0, 4.0]],
    "example2": [[5.5, 6.6, 7.7], [8.8, 9.9, 10.1], [11.2, 12.3, 13.4]],
}
# The source checker's other functional inputs.
SOURCE_FUNCTIONAL = {
    "all_zeros_4x4": lambda g: torch.zeros(4, 4),
    "identity_3x3": lambda g: torch.eye(3),
    "negative_values_2x2": lambda g: torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]),
    "large_N_16x16": lambda g: torch.empty(16, 16).uniform_(-10.0, 10.0, generator=g),
    "single_element": lambda g: torch.tensor([[42.0]]),
}


@pytest.mark.parametrize("name", SOURCE_EXAMPLES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_source_examples(backend, name):
    expected = torch.tensor(SOURCE_EXAMPLES[name], dtype=torch.float32)
    assert_same_bits(run(backend, expected.clone()), expected)


@pytest.mark.parametrize("name", SOURCE_FUNCTIONAL)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_source_functional_inputs(backend, name):
    a = SOURCE_FUNCTIONAL[name](torch.Generator().manual_seed(0)).float()
    assert_same_bits(run(backend, a.clone()), a)


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_special_values_bit_exact(backend, dtype):
    """Special patterns at every position of an odd-shaped matrix, beside random bits."""
    dtype = api.DTYPES[dtype]
    m, n = 7, 13
    special = from_bits(SPECIAL_BITS[dtype], dtype)
    generator = torch.Generator().manual_seed(0)
    info = torch.iinfo(BITS[dtype])
    random = torch.randint(info.min, info.max + 1, (m * n,), generator=generator)
    flat = random.to(BITS[dtype]).view(dtype)
    # Rotate the specials through every row and column, including the tails.
    for start in range(0, m * n, len(special) + 1):
        stop = min(start + len(special), m * n)
        flat[start:stop] = special[: stop - start]
    a = flat.reshape(m, n)
    assert_same_bits(run(backend, a.clone()), a)


@pytest.mark.parametrize("backend", api.BACKENDS)
def test_fresh_output(backend):
    """A new contiguous tensor with the input's shape, dtype and device; the input is
    unchanged."""
    a = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    x = a.to(device())
    out = run(backend, x)
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
    assert out.is_contiguous()
    assert out.data_ptr() != x.data_ptr()
    assert_same_bits(x, a)
    out.fill_(-1.0)  # writing the output must not reach the input
    assert_same_bits(x, a)


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_misaligned_input(backend, dtype):
    """A matrix starting one element into its storage is not 16-byte aligned: vector paths
    must fall back."""
    m, n = 33, 67
    full = Sampler(api.DTYPES[dtype], device()).randn(1 + m * n)
    a = full[1:].view(m, n)
    assert a.is_contiguous() and a.data_ptr() % 16 != 0
    assert_same_bits(run(backend, a), a)


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
@pytest.mark.parametrize("backend", [b for b in api.BACKENDS if b != "pytorch"])
def test_matches_reference(backend, case, dtype):
    check_against_reference(api, cases, backend, case, dtype)


CUDA_VARIANTS = {
    name: options["variant"] for name, (module, options) in api.VARIANTS.items() if module == "cuda"
}


def cuda_extension():
    skip_if_unavailable("cuda")
    from problems.matrix_copy.cuda.implementation import extension

    return extension()


@pytest.mark.parametrize("threads", [128, 512])
@pytest.mark.parametrize("shape", [(65, 129), (1025, 1025)], ids=str)
@pytest.mark.parametrize("backend", CUDA_VARIANTS)
def test_cuda_block_sizes(backend, shape, threads):
    """The 2-D block is (32, threads / 32) and the pack grids scale with threads."""
    a = Sampler(torch.float32, device()).randn(*shape)
    skip_if_unavailable(backend, api)
    assert_same_bits(api.matrix_copy(a, backend=backend, threads=threads), a)


@pytest.mark.parametrize("offsets", [(1, 0), (0, 1), (1, 1)], ids=["input", "output", "both"])
@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("backend", CUDA_VARIANTS)
def test_binding_misaligned_buffers(backend, dtype, offsets):
    """The api always allocates an aligned output, so a misaligned output can only reach a
    kernel through a direct call. Elements outside the output view must stay untouched."""
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    ext = cuda_extension()
    m, n = 33, 67
    a_offset, out_offset = offsets
    a = Sampler(dtype, "cuda").randn(1 + m * n)[a_offset : a_offset + m * n].view(m, n)
    out_buffer = torch.full((1 + m * n,), -7.0, dtype=dtype, device="cuda")
    out = out_buffer[out_offset : out_offset + m * n].view(m, n)
    assert (a.data_ptr() % 16 != 0) == bool(a_offset)
    assert (out.data_ptr() % 16 != 0) == bool(out_offset)
    ext.matrix_copy(a, out, CUDA_VARIANTS[backend], 256)
    assert_same_bits(out, a)
    guard = out_buffer[0] if out_offset else out_buffer[-1]
    assert guard.item() == -7.0


def test_binding_rejects_invalid_arguments():
    ext = cuda_extension()
    a = torch.zeros(4, 8, device="cuda")
    out = torch.empty_like(a)
    shared = torch.zeros(40, device="cuda")
    bad_calls = [
        (a, out, -1, 256),  # variant
        (a, out, 8, 256),
        (a, out, 0, 64),  # block size
        (a, out, 0, 1024),
        (a.flatten(), out.flatten(), 0, 256),  # 1-D
        (a.t(), torch.empty(8, 4, device="cuda"), 0, 256),  # non-contiguous input
        (a, torch.empty(8, 4, device="cuda"), 0, 256),  # shape mismatch
        (a, out.half(), 0, 256),  # dtype mismatch
        (a.double(), out.double(), 0, 256),  # fp64
        (a.cpu(), out.cpu(), 0, 256),  # host tensors
        (a, a, 0, 256),  # output is the input
        (shared[:32].view(4, 8), shared[8:].view(4, 8), 0, 256),  # partial overlap
    ]
    for args in bad_calls:
        with pytest.raises(RuntimeError):
            ext.matrix_copy(*args)


@pytest.mark.parametrize("tile", [(16, 128), (128, 16), (32, 32), (1, 1024)], ids=str)
def test_triton_tile_shapes(tile):
    skip_if_unavailable("triton")
    a = Sampler(torch.float16, device()).randn(65, 129)
    block_m, block_n = tile
    out = api.matrix_copy(a, backend="triton_2d", block_m=block_m, block_n=block_n)
    assert_same_bits(out, a)


def test_triton_rejects_invalid_options():
    skip_if_unavailable("triton")
    a = torch.zeros(4, 8, device="cuda")
    bad_options = [
        {"layout": "3d"},
        {"block_size": 1000},
        {"block_m": 0},
        {"block_n": 48},
        {"num_warps": 3},
    ]
    for options in bad_options:
        with pytest.raises(ValueError):
            api.matrix_copy(a, backend="triton", **options)


def test_rejects_invalid_inputs():
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    bad_tensors = [
        a.flatten(),  # 1-D
        a.reshape(1, 3, 4),  # 3-D
        a.t(),  # transposed view: 2-D but not contiguous
        a[:, ::2],  # strided columns
        a.double(),  # fp64
        a.int(),  # integer
    ]
    for bad in bad_tensors:
        with pytest.raises(ValueError):
            api.matrix_copy(bad)
    with pytest.raises(TypeError):
        api.matrix_copy([[1.0, 2.0], [3.0, 4.0]])


def test_rejects_unknown_backend():
    dtype = next(iter(api.DTYPES.values()))
    args = cases.make_inputs(cases.TEST_CASES[0], Sampler(dtype, "cpu"))
    with pytest.raises(ValueError, match="unknown backend"):
        api.matrix_copy(*args, backend="invalid")


def test_rejects_unknown_pytorch_method():
    a = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="unknown matrix_copy method"):
        api.matrix_copy(a, method="invalid")
