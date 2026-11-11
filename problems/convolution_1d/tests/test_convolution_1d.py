"""09 · 1D Convolution: every backend against the PyTorch reference, and the reference
against an independent FP64 oracle."""

import math

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
from problems.convolution_1d import api, cases

U32 = 2.0**-24  # FP32 unit roundoff


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
@pytest.mark.parametrize("backend", [b for b in api.BACKENDS if b != "pytorch"])
def test_matches_reference(backend, case, dtype):
    check_against_reference(api, cases, backend, case, dtype)


def test_rejects_unknown_backend():
    dtype = next(iter(api.DTYPES.values()))
    args = cases.make_inputs(cases.TEST_CASES[0], Sampler(dtype, "cpu"))
    with pytest.raises(ValueError, match="unknown backend"):
        api.conv1d(*args, backend="invalid")


# --- Reference against an FP64 oracle -------------------------------------------------


def oracle(x, w):
    """Exact sums by explicit loops: fsum of FP64 products. Inputs of every dtype convert
    to FP64 exactly, and a product of two FP32 values is exact in FP64 (48 <= 53 bits), so
    each result is the correctly rounded value of the exact sum.
    Returns (sums, sum of |products|) per output."""
    xs, ws = x.double().cpu().tolist(), w.double().cpu().tolist()
    k = len(ws)
    sums, mags = [], []
    for i in range(len(xs) - k + 1):
        products = [xs[i + j] * ws[j] for j in range(k)]
        sums.append(math.fsum(products))
        mags.append(math.fsum(abs(p) for p in products))
    return torch.tensor(sums, dtype=torch.float64), torch.tensor(mags, dtype=torch.float64)


def error_bound(exact, magnitude, k, dtype):
    """Componentwise bound for any FP32-accumulated dot product rounded once to `dtype`:
    |y_hat - y| <= u_d |y| + (1 + u_d) gamma_K sum_j |x_j w_j| + eta,
    with gamma_K = K u32 / (1 - K u32) (Higham, Thm 3.1; holds for any summation order,
    with or without FMA) and eta half the smallest subnormal of `dtype`. TF32 products or
    FP16 accumulation exceed it."""
    info = torch.finfo(dtype)
    u = info.eps / 2
    gamma = k * U32 / (1 - k * U32)
    eta = info.smallest_normal * info.eps / 2
    return u * exact.abs() + (1 + u) * gamma * magnitude + eta


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
def test_reference_matches_oracle(case, dtype):
    """On the GPU this is cuDNN. TF32 stays switched on globally (PyTorch's default) to
    show the reference turns it off for its own call."""
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    x, w = cases.make_inputs(case, Sampler(dtype, device(), seed=1))
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    try:
        out = api.conv1d(x, w)
        assert torch.backends.cudnn.allow_tf32  # restored after the call
    finally:
        torch.backends.cudnn.allow_tf32 = previous
    exact, magnitude = oracle(x, w)
    error = (out.double().cpu() - exact).abs()
    bound = error_bound(exact, magnitude, case["k"], dtype)
    worst = (error / bound).max().item()
    assert worst <= 1, f"error exceeds the FP32-accumulation bound by {worst:.2f}x"


def emulate(x, w, *, tf32=False, accumulate=torch.float32):
    """Sequential accumulation in a chosen order and precision, rounded once to x.dtype."""
    k, n_out = w.numel(), x.numel() - w.numel() + 1
    xf, wf = x.float(), w.float()
    if tf32:  # round to nearest, 10 explicit mantissa bits (13 FP32 bits dropped)
        xf, wf = (
            (t.view(torch.int32) + 0x1000).bitwise_and(~0x1FFF).view(torch.float32)
            for t in (xf, wf)
        )
    acc = torch.zeros(n_out, dtype=accumulate, device=x.device)
    for j in range(k):
        acc = (acc.float() + xf[j : j + n_out] * wf[j]).to(accumulate)
    return acc.to(x.dtype)


@pytest.mark.parametrize("k", [255, 1023, 2047])
@pytest.mark.parametrize("dtype", api.DTYPES)
def test_tolerances_cover_summation_order(k, dtype):
    """Any correct backend differs from the reference by summation order and one final
    rounding. A sequential FP32 sum must pass the fixed TOLERANCES at long filters, where
    the noise (about sqrt(K)) is largest."""
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    x, w = cases.make_inputs({"n": k + 4095, "k": k}, Sampler(dtype, device(), seed=2))
    rtol, atol = api.TOLERANCES[dtype]
    torch.testing.assert_close(emulate(x, w), api.conv1d(x, w), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "dtype, mode",
    [("fp32", {"tf32": True}), ("fp16", {"accumulate": torch.float16})],
    ids=["fp32-tf32-products", "fp16-fp16-accumulation"],
)
def test_tolerances_reject_reduced_precision(dtype, mode):
    """Negative controls: the tolerances are tight enough to catch the two likely
    precision shortcuts at the K = 255 bench filter."""
    dtype = api.DTYPES[dtype]
    x, w = cases.make_inputs({"n": 255 + 4095, "k": 255}, Sampler(dtype, device(), seed=3))
    rtol, atol = api.TOLERANCES[dtype]
    with pytest.raises(AssertionError):
        torch.testing.assert_close(emulate(x, w, **mode), api.conv1d(x, w), rtol=rtol, atol=atol)


# --- Contract -----------------------------------------------------------------------


@pytest.mark.parametrize("dtype", api.DTYPES.values())
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
def test_reference_contract(case, dtype):
    n, k = case["n"], case["k"]
    # Small integers: every partial sum is exact in FP32, so an FP32-accumulating result is
    # the exact sum rounded once to `dtype`, which is what `expected` holds. Reduced-precision
    # accumulation would round partial sums above 2048 (FP16) or 256 (BF16) and fail.
    # Both tensors start one element into their storage; that must be accepted.
    x = (torch.arange(n + 1) % 7 - 3).to(dtype)[1:]
    w = (torch.arange(k + 1) % 3 - 1).to(dtype)[1:]
    assert x.storage_offset() == 1 and w.storage_offset() == 1
    x_before, w_before = x.clone(), w.clone()
    out = api.conv1d(x, w)
    xs, ws = x.tolist(), w.tolist()
    expected = torch.tensor(
        [sum(xs[i + j] * ws[j] for j in range(k)) for i in range(n - k + 1)], dtype=dtype
    )
    assert out.shape == (n - k + 1,) and out.dtype == dtype and out.device == x.device
    assert out.is_contiguous() and out.storage_offset() == 0
    assert torch.equal(out, expected)
    out.fill_(-1)  # a fresh buffer: writing it cannot reach the inputs
    assert torch.equal(x, x_before) and torch.equal(w, w_before)


def test_reference_is_correlation_not_convolution():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w = torch.tensor([1.0, 0.0, -1.0])
    # out[i] = x[i] - x[i + 2]; a flipped filter would give the negation.
    assert torch.equal(api.conv1d(x, w), torch.tensor([-2.0, -2.0]))


@pytest.mark.parametrize("which", ["x", "w", "both"])
@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_misaligned_input(backend, dtype, which):
    """Inputs starting one element into their storage are not 16-byte aligned, so any
    vectorized or async-copy path must fall back. Compared with the reference on aligned
    copies of the same values."""
    skip_if_unavailable(backend, api)
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    n, k = 1031, 37
    s = Sampler(dtype, device(), seed=4)
    x_full, w_full = s.randn(n + 1), s.randn(k + 1)
    x = x_full[1:] if which in ("x", "both") else x_full[:n]
    w = w_full[1:] if which in ("w", "both") else w_full[:k]
    if which in ("x", "both"):
        assert x.data_ptr() % 16 != 0
    if which in ("w", "both"):
        assert w.data_ptr() % 16 != 0
    out = call_or_skip(api, backend, (x, w))
    expected = api.conv1d(x.clone(), w.clone())
    assert x.clone().data_ptr() % 16 == 0
    rtol, atol = api.TOLERANCES[dtype]
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "options",
    [{"block_size": 32, "num_warps": 1}, {"block_size": 128}, {"block_size": 4096, "num_warps": 8}],
    ids=str,
)
@pytest.mark.parametrize("case", [{"n": 1, "k": 1}, {"n": 5000, "k": 255}], ids=case_id)
def test_triton_tile_shapes(options, case):
    """Programs smaller than a warp, a single partial program, and many full ones."""
    skip_if_unavailable("triton", api)
    x, w = cases.make_inputs(case, Sampler(torch.float16, device(), seed=5))
    rtol, atol = api.TOLERANCES[torch.float16]
    out = api.conv1d(x, w, backend="triton", **options)
    torch.testing.assert_close(out, api.conv1d(x, w), rtol=rtol, atol=atol)


def test_triton_rejects_invalid_options():
    skip_if_unavailable("triton", api)
    x, w = torch.zeros(8, device="cuda"), torch.zeros(3, device="cuda")
    for options in [{"block_size": 0}, {"block_size": 48}, {"num_warps": 3}]:
        with pytest.raises(ValueError):
            api.conv1d(x, w, backend="triton", **options)


def test_rejects_invalid_inputs():
    x, w = torch.zeros(8), torch.zeros(3)
    bad = [
        (x[None], w),  # 2-D
        (x, w[None]),
        (x[::2], w),  # non-contiguous
        (x, torch.zeros(6)[::2]),
        (x, torch.zeros(9)),  # K > N
        (x, torch.zeros(0)),  # K = 0
        (torch.zeros(0), torch.zeros(0)),
        (x, w.half()),  # dtype mismatch
        (x.double(), w.double()),  # unsupported dtypes
        (x.int(), w.int()),
        (x.to_sparse(), w),
        (torch.zeros(8, requires_grad=True), w),
    ]
    for args in bad:
        with pytest.raises(ValueError):
            api.conv1d(*args)
    with pytest.raises(TypeError):
        api.conv1d([1.0, 2.0], w)
    if torch.cuda.is_available():
        with pytest.raises(ValueError):
            api.conv1d(torch.zeros(8, device="cuda"), w)


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_gpu_backends_reject_cpu(backend):
    with pytest.raises(ValueError, match="requires NVIDIA CUDA"):
        api.conv1d(torch.zeros(8), torch.zeros(3), backend=backend)


def test_traffic_model():
    c = {"n": 25_000_000, "k": 255}
    # x (N), w (K) and out (N - K + 1) once each: (2N + 1) elements.
    assert cases.logical_bytes(c, torch.float32) == 4 * (2 * 25_000_000 + 1)
    assert cases.logical_bytes(c, torch.float16) == 2 * (2 * 25_000_000 + 1)
    assert cases.flops(c) == 2 * 255 * (25_000_000 - 254)


# --- CUDA variants --------------------------------------------------------------------

CUDA_VARIANTS = {
    name: options["variant"] for name, (module, options) in api.VARIANTS.items() if module == "cuda"
}


def cuda_extension():
    skip_if_unavailable("cuda")
    from problems.convolution_1d.cuda.implementation import extension

    return extension()


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_cuda_const_capacity(dtype):
    """K = 16384 fills the 64 KiB __constant__ buffer exactly; one more tap is rejected.
    FP16/BF16 keep FP32 summation noise far below atol at this K (FP32 would not; see
    docs/design.md, "Tolerances")."""
    from problems.convolution_1d.cuda.implementation import MAX_CONST_TAPS

    ext = cuda_extension()
    assert ext.max_const_taps() == MAX_CONST_TAPS == 16384
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    s = Sampler(dtype, "cuda", seed=6)
    x, w = s.randn(MAX_CONST_TAPS + 777), s.randn(MAX_CONST_TAPS)
    rtol, atol = api.TOLERANCES[dtype]
    out = api.conv1d(x, w, backend="cuda_const")
    torch.testing.assert_close(out, api.conv1d(x, w), rtol=rtol, atol=atol)
    with pytest.raises(ValueError, match="16384"):
        api.conv1d(x, s.randn(MAX_CONST_TAPS + 1), backend="cuda_const")


def test_cuda_const_reloads_filter():
    """__constant__ memory persists between calls: each call must upload its own filter,
    including a shorter one after a longer one, with no host synchronization between."""
    skip_if_unavailable("cuda")
    s = Sampler(torch.float32, "cuda", seed=7)
    x = s.randn(4096)
    filters = [s.randn(255), s.randn(3), s.randn(255), s.randn(31)]
    outs = [api.conv1d(x, w, backend="cuda_const") for w in filters]
    rtol, atol = api.TOLERANCES[torch.float32]
    for w, out in zip(filters, outs, strict=True):
        torch.testing.assert_close(out, api.conv1d(x, w), rtol=rtol, atol=atol)


def test_cuda_rejects_invalid_variant():
    skip_if_unavailable("cuda")
    x, w = torch.zeros(8, device="cuda"), torch.zeros(3, device="cuda")
    with pytest.raises(ValueError):
        api.conv1d(x, w, backend="cuda", variant=2)


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("backend", CUDA_VARIANTS)
def test_binding_misaligned_output(backend, dtype):
    """The api always allocates an aligned output, so a misaligned one needs a direct call.
    Elements outside the output view must stay untouched."""
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    ext = cuda_extension()
    n, k = 1031, 37
    s = Sampler(dtype, "cuda", seed=8)
    x, w = s.randn(n), s.randn(k)
    buffer = torch.full((n - k + 3,), -7.0, dtype=dtype, device="cuda")
    out = buffer[1 : n - k + 2]
    assert out.data_ptr() % 16 != 0
    ext.conv1d(x, w, out, CUDA_VARIANTS[backend])
    rtol, atol = api.TOLERANCES[dtype]
    torch.testing.assert_close(out, api.conv1d(x, w), rtol=rtol, atol=atol)
    assert buffer[0].item() == -7.0 and buffer[-1].item() == -7.0


def test_binding_rejects_invalid_arguments():
    ext = cuda_extension()
    x, w = torch.zeros(8, device="cuda"), torch.zeros(3, device="cuda")
    out = torch.empty(6, device="cuda")
    shared = torch.zeros(16, device="cuda")
    bad_calls = [
        (x, w, out, -1),  # variant
        (x, w, out, 2),
        (x, w, torch.empty(5, device="cuda"), 0),  # wrong output length
        (x, w, out.half(), 0),  # dtype mismatch
        (x, w.half(), out, 0),
        (x.double(), w.double(), out.double(), 0),  # fp64
        (x.cpu(), w.cpu(), out.cpu(), 0),  # host tensors
        (x[None], w, out, 0),  # 2-D
        (torch.zeros(16, device="cuda")[::2], w, out, 0),  # non-contiguous
        (x, torch.zeros(9, device="cuda"), torch.empty(0, device="cuda"), 0),  # K > N
        (shared[:8], w, shared[2:8], 0),  # output overlaps x
        (x, shared[:3], shared[1:7], 1),  # output overlaps w
    ]
    for args in bad_calls:
        with pytest.raises(RuntimeError):
            ext.conv1d(*args)
