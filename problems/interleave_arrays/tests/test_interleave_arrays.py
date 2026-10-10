"""04 · Interleave Arrays: source examples, output ordering, every backend vs the reference."""

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
from problems.interleave_arrays import api, cases

# The statement's worked examples, plus inputs in the spirit of its functional cases
# (single element, negatives, mixed signs, zeros and ones, extreme magnitudes). FP32.
SOURCE_EXAMPLES = {
    "example1": ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    "example2": ([10.0, 20.0], [30.0, 40.0]),
    "single": ([7.0], [-7.0]),
    "negatives": ([-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]),
    "mixed_signs": ([1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]),
    "zeros_ones": ([0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0]),
    "extremes": ([1e10, -1e-10], [-1e10, 1e-10]),
}


def interleaved(a, b):
    """Plain-Python oracle, independent of PyTorch indexing."""
    return [v for pair in zip(a, b, strict=True) for v in pair]


@pytest.mark.parametrize("name", SOURCE_EXAMPLES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_source_examples(backend, name):
    skip_if_unavailable(backend, api)
    a, b = SOURCE_EXAMPLES[name]
    args = (torch.tensor(a, device=device()), torch.tensor(b, device=device()))
    actual = call_or_skip(api, backend, args)
    torch.testing.assert_close(actual.cpu(), torch.tensor(interleaved(a, b)), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("n", [1, 5, 17, 33, 128])
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_output_order(backend, n, dtype):
    """a = 0, 2, 4, ... and b = 1, 3, 5, ... must interleave to exactly 0, 1, 2, ..."""
    skip_if_unavailable(backend, api)
    skip_unsupported_dtype(api.DTYPES[dtype])
    args = cases.make_inputs({"n": n, "dist": "ordered"}, Sampler(api.DTYPES[dtype], device()))
    actual = call_or_skip(api, backend, args)
    expected = torch.arange(2 * n, dtype=torch.float32).to(api.DTYPES[dtype])
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend", api.BACKENDS)
def test_fresh_output(backend):
    """A new 2N-element tensor of the input dtype and device, sharing no storage."""
    skip_if_unavailable(backend, api)
    a, b = cases.make_inputs({"n": 1025, "dist": "normal"}, Sampler(torch.float32, device()))
    out = call_or_skip(api, backend, (a, b))
    assert out.shape == (2050,) and out.dtype == a.dtype and out.device == a.device
    assert out.is_contiguous()
    for x in (a, b):
        start, end = x.data_ptr(), x.data_ptr() + x.nbytes
        assert not (out.data_ptr() < end and start < out.data_ptr() + out.nbytes)


@pytest.mark.parametrize("backend", api.BACKENDS)
def test_same_input_twice(backend):
    skip_if_unavailable(backend, api)
    a = torch.tensor([1.0, 2.0, 3.0], device=device())
    actual = call_or_skip(api, backend, (a, a))
    torch.testing.assert_close(actual.cpu(), torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
@pytest.mark.parametrize("backend", [b for b in api.BACKENDS if b != "pytorch"])
def test_matches_reference(backend, case, dtype):
    check_against_reference(api, cases, backend, case, dtype)


def test_rejects_invalid_inputs():
    x = torch.ones(8)
    invalid = [
        (x, x[:7]),  # different lengths
        (x, x.half()),  # different dtypes
        (x.view(2, 4), x.view(2, 4)),  # not 1-D
        (x[::2], x[::2]),  # not contiguous
        (x.double(), x.double()),  # unsupported dtype
        (torch.ones(8, requires_grad=True), x),  # autograd
    ]
    for a, b in invalid:
        with pytest.raises(ValueError):
            api.interleave(a, b)
    with pytest.raises(TypeError):
        api.interleave([1.0], x)


def test_rejects_unknown_backend():
    args = cases.make_inputs(cases.TEST_CASES[0], Sampler(torch.float32, "cpu"))
    with pytest.raises(ValueError, match="unknown backend"):
        api.interleave(*args, backend="invalid")
