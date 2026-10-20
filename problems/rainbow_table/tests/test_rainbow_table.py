"""06 · Rainbow Table: source examples, an independent Python oracle, every backend vs the
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
)
from problems.rainbow_table import api, cases

# torch.compile is exercised only on the GPU: on the CPU its generated C++ would rely on
# signed int32 overflow (undefined behaviour) and needs a C++ toolchain.
COMPILE_BACKENDS = {"pytorch_compile"}


def skip_compile_without_gpu(backend):
    if backend in COMPILE_BACKENDS and device() != "cuda":
        pytest.skip("torch.compile baseline is tested on the GPU only")


def fnv1a(value):
    """Plain-Python FNV-1a over the 4 bytes of a 32-bit pattern, least significant first."""
    h = 2166136261
    for k in range(4):
        h = ((h ^ ((value >> (8 * k)) & 0xFF)) * 16777619) & 0xFFFFFFFF
    return h


def python_hashes(values, rounds):
    out = []
    for v in values:
        v &= 0xFFFFFFFF  # the int32 bit pattern of a negative input
        for _ in range(rounds):
            v = fnv1a(v)
        out.append(v)
    return out


def run(backend, values, rounds):
    x = torch.tensor(values, dtype=torch.int32, device=device())
    return call_or_skip(api, backend, (x, rounds))


# The statement's worked examples, with its published outputs.
SOURCE_EXAMPLES = {
    "example1": ([123, 456, 789], 2, [1636807824, 1273011621, 2193987222]),
    "example2": ([0, 1, 2147483647], 3, [96754810, 3571711400, 2006156166]),
}
# The statement's other functional inputs; expected values come from the Python oracle.
SOURCE_FUNCTIONAL = {
    "single": ([100], 1),
    "pair": ([100, 200], 1),
    "single_round": ([1, 2, 3, 4, 5], 1),
}
EDGE_VALUES = [0, 1, 255, 256, 65535, 65536, 2**24, cases.INT32_MAX, -1, -256, cases.INT32_MIN]


@pytest.mark.parametrize("name", SOURCE_EXAMPLES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_source_examples(backend, name):
    skip_compile_without_gpu(backend)
    skip_if_unavailable(backend, api)
    values, rounds, expected = SOURCE_EXAMPLES[name]
    assert python_hashes(values, rounds) == expected  # the oracle agrees with the source
    assert run(backend, values, rounds).tolist() == expected


@pytest.mark.parametrize("name", SOURCE_FUNCTIONAL)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_source_functional_inputs(backend, name):
    skip_compile_without_gpu(backend)
    skip_if_unavailable(backend, api)
    values, rounds = SOURCE_FUNCTIONAL[name]
    assert run(backend, values, rounds).tolist() == python_hashes(values, rounds)


@pytest.mark.parametrize("rounds", [0, 1, 2, 7, 101])
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_python_oracle(backend, rounds):
    """Edge values and random int32 patterns against plain Python, independent of PyTorch."""
    skip_compile_without_gpu(backend)
    skip_if_unavailable(backend, api)
    generator = torch.Generator().manual_seed(rounds)
    random = torch.randint(cases.INT32_MIN, cases.INT32_MAX + 1, (300,), generator=generator)
    values = EDGE_VALUES + random.tolist()
    assert run(backend, values, rounds).tolist() == python_hashes(values, rounds)


@pytest.mark.parametrize("backend", api.BACKENDS)
def test_fresh_output(backend):
    """A new contiguous uint32 tensor of the same length and device; the input is unchanged."""
    skip_compile_without_gpu(backend)
    skip_if_unavailable(backend, api)
    for rounds in (0, 3):
        x = torch.arange(-3, 6, dtype=torch.int32, device=device())
        before = x.clone()
        out = call_or_skip(api, backend, (x, rounds))
        assert out.shape == x.shape and out.dtype == torch.uint32 and out.device == x.device
        assert out.is_contiguous()
        assert out.data_ptr() != x.data_ptr()  # even R = 0 must not return a view of x
        torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize("backend", api.BACKENDS)
def test_misaligned_input(backend):
    """A slice starting one element in is not 16-byte aligned: vector paths must fall back."""
    skip_compile_without_gpu(backend)
    skip_if_unavailable(backend, api)
    (full, _) = cases.make_inputs(
        {"n": 41, "rounds": 3, "dist": "full"}, Sampler(torch.int32, device())
    )
    x = full[1:]
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    assert call_or_skip(api, backend, (x, 3)).tolist() == python_hashes(x.tolist(), 3)


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
@pytest.mark.parametrize("backend", [b for b in api.BACKENDS if b != "pytorch"])
def test_matches_reference(backend, case, dtype):
    skip_compile_without_gpu(backend)
    check_against_reference(api, cases, backend, case, dtype)


def test_rejects_invalid_inputs():
    x = torch.arange(8, dtype=torch.int32)
    bad_tensors = [
        x.float(),  # float
        x.long(),  # int64
        x.view(torch.uint32),  # the output dtype, not the input's
        x.reshape(2, 4),  # 2-D
        x[::2],  # strided
    ]
    for bad in bad_tensors:
        with pytest.raises(ValueError):
            api.rainbow_table(bad, 1)
    with pytest.raises(ValueError):
        api.rainbow_table(x, -1)
    for bad_rounds in (1.0, True, "2", None):
        with pytest.raises(TypeError):
            api.rainbow_table(x, bad_rounds)
    with pytest.raises(TypeError):
        api.rainbow_table([1, 2, 3], 1)


def test_rejects_unknown_backend():
    args = cases.make_inputs(cases.TEST_CASES[0], Sampler(torch.int32, "cpu"))
    with pytest.raises(ValueError, match="unknown backend"):
        api.rainbow_table(*args, backend="invalid")


def test_rejects_unknown_pytorch_method():
    x = torch.arange(4, dtype=torch.int32)
    with pytest.raises(ValueError, match="unknown rainbow_table method"):
        api.rainbow_table(x, 1, method="invalid")
