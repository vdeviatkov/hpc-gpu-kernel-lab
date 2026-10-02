import importlib
import importlib.util

import pytest
import torch

from problems.vector_add import api

CUDA = torch.cuda.is_available() and torch.version.cuda is not None


@pytest.fixture(params=api.BACKENDS)
def backend(request):
    name = request.param
    if name != "pytorch" and not CUDA:
        pytest.skip("NVIDIA GPU unavailable")
    if name == "triton" and importlib.util.find_spec("triton") is None:
        pytest.skip("optional Triton package unavailable")
    if name.startswith("cuda_"):
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            pytest.skip("CUDA toolkit unavailable")
    return name


@pytest.mark.parametrize("dtype", api.DTYPES.values())
@pytest.mark.parametrize("n", [0, 1, 3, 4, 5, 31, 127, 255, 256, 257, 1025, 1048583])
@pytest.mark.parametrize("offset", [0, 1, 3])
def test_values_and_guards(backend, dtype, n, offset):
    if CUDA and dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 test policy requires SM80+")
    device = "cuda" if CUDA else "cpu"
    torch.manual_seed(42)
    a = torch.randn(n + offset, device=device, dtype=dtype)[offset:]
    b = torch.randn(n + offset, device=device, dtype=dtype)[offset:]
    before_a, before_b = a.clone(), b.clone()
    backing = torch.full((n + offset + 7,), -91, device=device, dtype=dtype)
    out = backing[offset : offset + n]
    result = api.add(a, b, backend=backend, out=out)
    assert result is out
    rtol, atol = api.TOLERANCES[dtype]
    torch.testing.assert_close(result, (a.float() + b.float()).to(dtype), rtol=rtol, atol=atol)
    torch.testing.assert_close(a, before_a, rtol=0, atol=0)
    torch.testing.assert_close(b, before_b, rtol=0, atol=0)
    assert torch.all(backing[:offset] == -91)
    assert torch.all(backing[offset + n :] == -91)


def test_special_values_and_allocation(backend):
    device = "cuda" if CUDA else "cpu"
    a = torch.tensor([0.0, -0.0, 1e20, -1e20, float("inf"), float("nan"), -3.0], device=device)
    b = torch.tensor([-0.0, -0.0, -1e20, 1.0, -float("inf"), 1.0, 3.0], device=device)
    result = api.add(a, b, backend=backend)
    torch.testing.assert_close(result, a + b, equal_nan=True, rtol=0, atol=0)
    assert result.data_ptr() not in (a.data_ptr(), b.data_ptr())
    torch.testing.assert_close(api.add(a, a, backend=backend), a + a, equal_nan=True)


@pytest.mark.parametrize("dtype", api.DTYPES.values())
@pytest.mark.parametrize("output_offset", [0, 1, 2, 3, 8])
def test_output_alignment_and_overlap(backend, dtype, output_offset):
    if CUDA and dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 test policy requires SM80+")
    device = "cuda" if CUDA else "cpu"
    a = torch.arange(1031, device=device, dtype=dtype)
    b = a.clone()
    backing = torch.full((1042,), -91.0, device=device, dtype=dtype)
    out = backing[output_offset : output_offset + a.numel()]
    api.add(a, b, out=out, backend=backend)
    torch.testing.assert_close(out, a + b)
    assert torch.all(backing[:output_offset] == -91)
    assert torch.all(backing[output_offset + a.numel() :] == -91)
    with pytest.raises(ValueError, match="overlap"):
        api.add(a[:-1], b[:-1], out=a[1:], backend=backend)


@pytest.mark.parametrize("dtype", api.DTYPES.values())
def test_grid_stride_second_iteration(backend, dtype):
    if CUDA and dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 test policy requires SM80+")
    device = "cuda" if CUDA else "cpu"
    # Exceed 4096 blocks * 256 threads * one 16-byte pack (eight FP16/BF16 elements),
    # including a partial-pack tail. Values stay exactly representable in BF16.
    a = (torch.arange(4096 * 256 * 8 + 11, device=device) % 128).to(dtype)
    torch.testing.assert_close(api.add(a, a, backend=backend), a + a, rtol=0, atol=0)


def test_disjoint_storage(backend):
    device = "cuda" if CUDA else "cpu"
    backing = torch.arange(300, device=device, dtype=torch.float32)
    a, b, out = backing[:100], backing[100:200], backing[200:]
    expected = a + b
    api.add(a, b, out=out, backend=backend)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("threads", [128, 512])
@pytest.mark.parametrize("block_size,num_warps", [(256, 4), (2048, 8)])
def test_launch_configuration(backend, threads, block_size, num_warps):
    device = "cuda" if CUDA else "cpu"
    a = torch.arange(8199, device=device, dtype=torch.float32)
    actual = api.add(
        a, a, backend=backend, threads=threads, block_size=block_size, num_warps=num_warps
    )
    torch.testing.assert_close(actual, a + a)


def test_reject_invalid_contracts():
    a = torch.ones(8)
    invalid = [
        (a, a[:3], None),
        (a, a.double(), None),
        (a.view(2, 4), a.view(2, 4), None),
        (a[::2], a[::2], None),
        (a.clone().requires_grad_(), a, None),
    ]
    for x, y, out in invalid:
        with pytest.raises(ValueError):
            api.add(x, y, out=out)
    a = a.detach()
    for out in (a, a[1:]):
        with pytest.raises(ValueError):
            api.add(a[: out.numel()], a[: out.numel()], out=out)
    with pytest.raises(ValueError, match="unknown backend"):
        api.add(a, a, backend="invalid")
    with pytest.raises(ValueError, match="NVIDIA"):
        api.add(a, a, backend="cuda_scalar")


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="NVIDIA GPU unavailable")
def test_nondefault_stream(backend):
    # Build on the default stream, then produce inputs and consume output entirely
    # on a nondefault stream. Synchronize only that stream at the boundary.
    a, b = torch.ones(1025, device="cuda"), torch.ones(1025, device="cuda")
    api.add(a, b, backend=backend)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a.fill_(7)
        b.fill_(-2)
        result = api.add(a, b, backend=backend) * 3
    stream.synchronize()
    torch.testing.assert_close(result, torch.full_like(result, 15))


@pytest.mark.jax
@pytest.mark.parametrize("dtype_name", api.DTYPES)
@pytest.mark.parametrize("n", [0, 1, 1025])
def test_jax(dtype_name, n):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from problems.vector_add.jax import implementation as impl

    dtype = api.DTYPES[dtype_name]
    a = torch.linspace(-3, 4, n).to(dtype)
    b = torch.linspace(2, -1, n).to(dtype)
    jd = {"fp32": jnp.float32, "fp16": jnp.float16, "bf16": jnp.bfloat16}[dtype_name]
    ja, jb = jnp.asarray(a.float().numpy(), dtype=jd), jnp.asarray(b.float().numpy(), dtype=jd)
    result = jax.device_get(impl.add(ja, jb).block_until_ready()).astype("float32")
    rtol, atol = api.TOLERANCES[dtype]
    torch.testing.assert_close(torch.from_numpy(result), (a + b).float(), rtol=rtol, atol=atol)
