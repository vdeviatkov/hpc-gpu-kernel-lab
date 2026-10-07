"""02 · ReLU: the source examples, then every backend against the PyTorch reference."""

import pytest
import torch

from lab.data import Sampler
from lab.testing import call_or_skip, case_id, check_against_reference, device, skip_if_unavailable
from problems.relu import api, cases

# The worked examples from the LeetGPU statement.
SOURCE_EXAMPLES = [
    ([-2.0, -1.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0, 2.0]),
    ([-3.5, 0.0, 4.2], [0.0, 0.0, 4.2]),
]


@pytest.mark.parametrize("example", SOURCE_EXAMPLES, ids=["example1", "example2"])
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_source_examples(backend, example):
    skip_if_unavailable(backend, api)
    values, expected = example
    x = torch.tensor(values, device=device())
    actual = call_or_skip(api, backend, (x,))
    torch.testing.assert_close(actual.cpu(), torch.tensor(expected), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
@pytest.mark.parametrize("backend", [b for b in api.BACKENDS if b != "pytorch"])
def test_matches_reference(backend, case, dtype):
    check_against_reference(api, cases, backend, case, dtype)


@pytest.mark.parametrize("dist", ["alternating", "warp_blocks"])
def test_sign_patterns(dist):
    (x,) = cases.make_inputs({"n": 64, "dist": dist}, Sampler(torch.float32, "cpu"))
    period = 1 if dist == "alternating" else 32
    expected_negative = (torch.arange(64) // period) % 2 == 1
    assert torch.equal(x < 0, expected_negative)


def test_rejects_invalid_inputs():
    x = torch.ones(8)
    for bad in (x.view(2, 4), x[::2], x.double()):
        with pytest.raises(ValueError):
            api.relu(bad)
    with pytest.raises(TypeError):
        api.relu([1.0, 2.0])


def test_rejects_unknown_backend():
    dtype = next(iter(api.DTYPES.values()))
    args = cases.make_inputs(cases.TEST_CASES[0], Sampler(dtype, "cpu"))
    with pytest.raises(ValueError, match="unknown backend"):
        api.relu(*args, backend="invalid")
