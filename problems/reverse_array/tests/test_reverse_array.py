"""03 · Reverse Array: source examples, in-place semantics, every backend vs the reference."""

import pytest
import torch

from lab.data import Sampler
from lab.testing import (
    call_or_skip,
    case_id,
    check_against_reference,
    device,
    skip_if_unavailable,
)
from problems.reverse_array import api, cases

# The worked examples and the explicit functional cases from the LeetGPU statement, plus
# an odd length to pin the middle element. Each input reversed is the expected output.
SOURCE_EXAMPLES = {
    "example1": [1.0, 2.0, 3.0, 4.0],
    "example2": [1.5, 2.5, 3.5],
    "single": [42.0],
    "negatives": [-1.0, -2.0, -3.0, -4.0],
    "mixed_signs": [1.0, -2.0, 3.0, -4.0],
    "tiny": [1e-6, 1e-7, 1e-8, 1e-9],
    "large": [1e6, 1e7, -1e6, -1e7],
    "sequential": [float(i) for i in range(30)],
    "odd_middle": [1.0, 2.0, 3.0, 4.0, 5.0],
}


@pytest.mark.parametrize("name", SOURCE_EXAMPLES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_source_examples(backend, name):
    skip_if_unavailable(backend, api)
    values = SOURCE_EXAMPLES[name]
    x = torch.tensor(values, device=device())
    actual = call_or_skip(api, backend, (x,))
    torch.testing.assert_close(actual.cpu(), torch.tensor(values[::-1]), rtol=0, atol=0)


@pytest.mark.parametrize("n", [1, 2, 7, 1025])
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_reverses_in_place(backend, n):
    """The result is the input tensor itself: same storage, now reversed."""
    skip_if_unavailable(backend, api)
    x = torch.arange(n, dtype=torch.float32, device=device())
    pointer = x.data_ptr()
    result = call_or_skip(api, backend, (x,))
    assert result is x
    assert x.data_ptr() == pointer
    torch.testing.assert_close(x.cpu(), torch.arange(n - 1, -1, -1, dtype=torch.float32))


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
@pytest.mark.parametrize("backend", [b for b in api.BACKENDS if b != "pytorch"])
def test_matches_reference(backend, case, dtype):
    check_against_reference(api, cases, backend, case, dtype)


def test_values_stay_finite():
    for dtype in api.DTYPES.values():
        (x,) = cases.make_inputs({"n": 1000, "dist": "large"}, Sampler(dtype, "cpu"))
        assert torch.isfinite(x).all()


def test_rejects_invalid_inputs():
    x = torch.ones(8)
    for bad in (x.view(2, 4), x[::2], x.double(), torch.ones(8, requires_grad=True)):
        with pytest.raises(ValueError):
            api.reverse_(bad)
    with pytest.raises(TypeError):
        api.reverse_([1.0, 2.0])


def test_rejects_unknown_backend():
    dtype = next(iter(api.DTYPES.values()))
    args = cases.make_inputs(cases.TEST_CASES[0], Sampler(dtype, "cpu"))
    with pytest.raises(ValueError, match="unknown backend"):
        api.reverse_(*args, backend="invalid")
