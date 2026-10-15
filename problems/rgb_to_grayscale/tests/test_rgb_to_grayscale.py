"""05 · RGB to Grayscale: source examples, channel order, every backend vs the reference."""

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
from problems.rgb_to_grayscale import api, cases

# torch.compile is exercised only on the GPU: on CPU it needs a C++ toolchain and slows the
# CPU-only CI for no extra coverage of this problem's kernels.
COMPILE_BACKENDS = {"pytorch_compile"}


def skip_compile_without_gpu(backend):
    if backend in COMPILE_BACKENDS and device() != "cuda":
        pytest.skip("torch.compile baseline is tested on the GPU only")


# The statement's worked examples: (height, width, flat RGB input, expected gray).
SOURCE_EXAMPLES = {
    "example1": (
        2,
        2,
        [255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128],
        [76.245, 149.685, 29.07, 128.0],
    ),
    "example2": (1, 1, [100, 150, 200], [140.75]),
    "all_zeros": (2, 2, [0] * 12, [0.0] * 4),
    "all_255": (2, 2, [255] * 12, [255.0] * 4),
}
COEFFICIENTS = (0.299, 0.587, 0.114)


@pytest.mark.parametrize("name", SOURCE_EXAMPLES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_source_examples(backend, name):
    skip_compile_without_gpu(backend)
    skip_if_unavailable(backend, api)
    h, w, values, expected = SOURCE_EXAMPLES[name]
    image = torch.tensor(values, dtype=torch.float32, device=device()).reshape(h, w, 3)
    actual = call_or_skip(api, backend, (image,))
    # The source's own tolerance, against its published values.
    torch.testing.assert_close(
        actual.cpu(), torch.tensor(expected).reshape(h, w), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_channel_order(backend, dtype):
    """Pure red, green and blue pixels: a swapped channel gives a different coefficient."""
    skip_compile_without_gpu(backend)
    skip_if_unavailable(backend, api)
    skip_unsupported_dtype(api.DTYPES[dtype])
    image = torch.tensor(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=torch.float32, device=device()
    ).to(api.DTYPES[dtype])
    actual = call_or_skip(api, backend, (image,))
    expected = torch.tensor([[255 * c for c in COEFFICIENTS]])
    rtol, atol = api.TOLERANCES[api.DTYPES[dtype]]
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("backend", api.BACKENDS)
def test_fresh_output(backend):
    """A new contiguous (H, W) tensor of the image dtype and device."""
    skip_compile_without_gpu(backend)
    skip_if_unavailable(backend, api)
    (image,) = cases.make_inputs({"h": 5, "w": 7, "dist": "int"}, Sampler(torch.float32, device()))
    out = call_or_skip(api, backend, (image,))
    assert out.shape == (5, 7) and out.dtype == image.dtype and out.device == image.device
    assert out.is_contiguous()
    start, end = image.data_ptr(), image.data_ptr() + image.nbytes
    assert not (out.data_ptr() < end and start < out.data_ptr() + out.nbytes)


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_misaligned_image(backend, dtype):
    """A row slice of a larger image starts mid-allocation: vector paths must fall back."""
    skip_compile_without_gpu(backend)
    skip_if_unavailable(backend, api)
    skip_unsupported_dtype(api.DTYPES[dtype])
    sampler = Sampler(api.DTYPES[dtype], device())
    (full,) = cases.make_inputs({"h": 9, "w": 13, "dist": "int"}, sampler)
    image = full[1:]  # starts 13 * 3 elements in: not 16-byte aligned for any dtype here
    assert image.is_contiguous() and image.data_ptr() % 16 != 0
    actual = call_or_skip(api, backend, (image,))
    expected = call_or_skip(api, "pytorch", (image,))
    rtol, atol = api.TOLERANCES[api.DTYPES[dtype]]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
@pytest.mark.parametrize("backend", [b for b in api.BACKENDS if b != "pytorch"])
def test_matches_reference(backend, case, dtype):
    skip_compile_without_gpu(backend)
    check_against_reference(api, cases, backend, case, dtype)


def test_rejects_invalid_inputs():
    image = torch.ones(4, 5, 3)
    invalid = [
        torch.ones(4, 5, 4),  # not 3 channels
        torch.ones(20, 3),  # not (H, W, 3)
        torch.ones(3, 4, 5).permute(1, 2, 0),  # channels not interleaved (not contiguous)
        image[:, ::2],  # strided
        image.double(),  # unsupported dtype
        torch.ones(4, 5, 3, requires_grad=True),  # autograd
    ]
    for bad in invalid:
        with pytest.raises(ValueError):
            api.rgb_to_grayscale(bad)
    with pytest.raises(TypeError):
        api.rgb_to_grayscale([[[1.0, 2.0, 3.0]]])


def test_rejects_unknown_backend():
    args = cases.make_inputs(cases.TEST_CASES[0], Sampler(torch.float32, "cpu"))
    with pytest.raises(ValueError, match="unknown backend"):
        api.rgb_to_grayscale(*args, backend="invalid")
