"""10 · Gaussian Blur: every backend against the PyTorch reference, the reference against
an independent FP64 oracle, and the separable factorization against both."""

import math

import pytest
import torch
import torch.nn.functional as F

from lab import dispatch
from lab.data import Sampler
from lab.testing import (
    call_or_skip,
    case_id,
    check_against_reference,
    device,
    skip_if_unavailable,
    skip_unsupported_dtype,
)
from problems.gaussian_blur import api, cases

U32 = 2.0**-24  # FP32 unit roundoff
GAUSSIAN_CASES = [c for c in cases.TEST_CASES if c["kernel"] == "gaussian"]
RANDOM_CASES = [c for c in cases.TEST_CASES if c["kernel"] == "random"]


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
@pytest.mark.parametrize("backend", [b for b in api.BACKENDS if b != "pytorch"])
def test_matches_reference(backend, case, dtype):
    check_against_reference(api, cases, backend, case, dtype)


@pytest.mark.parametrize("backend", [b for b in api.BACKENDS if b != "pytorch"])
def test_source_performance_case(backend):
    """The source's timed case, at its own scale: 512 x 512 FP32 in [0, 255), 7 x 7
    unnormalized random kernel."""
    skip_if_unavailable(backend, api)
    s = Sampler(torch.float32, device(), seed=11)
    image, kernel = s.uniform(0, 255, 512, 512), s.uniform(0.0001, 0.02, 7, 7)
    out = call_or_skip(api, backend, (image, kernel))
    rtol, atol = api.TOLERANCES[torch.float32]
    torch.testing.assert_close(out, api.gaussian_blur(image, kernel), rtol=rtol, atol=atol)


def test_rejects_unknown_backend():
    dtype = next(iter(api.DTYPES.values()))
    args = cases.make_inputs(cases.TEST_CASES[0], Sampler(dtype, "cpu"))
    with pytest.raises(ValueError, match="unknown backend"):
        api.gaussian_blur(*args, backend="invalid")


# --- FP64 oracle ----------------------------------------------------------------------


def windows(image, kh, kw):
    """(H*W, KH*KW) zero-padded neighbourhoods in FP64: a pure gather, so exact."""
    h, w = image.shape
    x = image.detach().to("cpu", torch.float64).view(1, 1, h, w)
    return F.unfold(x, (kh, kw), padding=(kh // 2, kw // 2))[0].T


def oracle(image, kernel):
    """Exact results: fsum of FP64 products per pixel. Inputs of every dtype convert to
    FP64 exactly and a product of two FP32 values is exact in FP64 (48 <= 53 bits), so
    each value is the correctly rounded exact sum. Returns (sums, sum of |products|)."""
    (h, w), (kh, kw) = image.shape, kernel.shape
    products = windows(image, kh, kw) * kernel.detach().to("cpu", torch.float64).reshape(1, -1)
    sums = torch.tensor([math.fsum(p) for p in products.tolist()], dtype=torch.float64)
    return sums.view(h, w), products.abs().sum(1).view(h, w)


def gamma(n):
    return n * U32 / (1 - n * U32)


def error_bound(exact, magnitude, n, dtype):
    """Componentwise bound for any FP32-accumulated sum of n products rounded once to
    `dtype`: u_d |y| + (1 + u_d) gamma_n sum |x k| + eta (Higham, Thm 3.1; any summation
    order, with or without FMA), eta half the smallest subnormal of `dtype`. The term
    count n is KH * KW for the direct filter, so the bound grows with the kernel area."""
    info = torch.finfo(dtype)
    u = info.eps / 2
    eta = info.smallest_normal * info.eps / 2
    return u * exact.abs() + (1 + u) * gamma(n) * magnitude + eta


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
def test_reference_matches_oracle(case, dtype):
    """On the GPU this is cuDNN. TF32 stays switched on globally (PyTorch's default and the
    source checker's setting) to show the reference turns it off for its own call. Cases
    include 1 x 1 images, kernels larger than the image, borders where up to KH//2 rows
    and KW//2 columns of a window are padding, and odd W."""
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    image, kernel = cases.make_inputs(case, Sampler(dtype, device(), seed=1))
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    try:
        out = api.gaussian_blur(image, kernel)
        assert torch.backends.cudnn.allow_tf32  # restored after the call
    finally:
        torch.backends.cudnn.allow_tf32 = previous
    exact, magnitude = oracle(image, kernel)
    error = (out.double().cpu() - exact).abs()
    bound = error_bound(exact, magnitude, case["kh"] * case["kw"], dtype)
    worst = (error / bound).max().item()
    assert worst <= 1, f"error exceeds the FP32-accumulation bound by {worst:.2f}x"


# --- Separability ---------------------------------------------------------------------


def two_pass(image, col, row, *, intermediate=torch.float32):
    """The separable algorithm a backend would run: a KW-tap row pass into an
    intermediate, then a KH-tap column pass, both FP32-accumulated with FP32 factors
    (TF32 off), rounded once to the image dtype."""
    (h, w), kh, kw = image.shape, col.numel(), row.numel()
    x = image.float().view(1, 1, h, w)
    col, row = col.to(x.device), row.to(x.device)
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        t = F.conv2d(x, row.view(1, 1, 1, kw), padding=(0, kw // 2))
        t = t.to(intermediate).float()
        out = F.conv2d(t, col.view(1, 1, kh, 1), padding=(kh // 2, 0))
    return out.view(h, w).to(image.dtype)


def residual(kernel, col, row):
    """kernel - outer(col, row), exactly in FP64."""
    return kernel.detach().to("cpu", torch.float64) - torch.outer(col.double(), row.double())


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", GAUSSIAN_CASES, ids=case_id)
def test_gaussians_are_separable(case, dtype):
    """Every Gaussian kernel, rounded to every dtype, is accepted, and its factors
    reproduce it within SEPARABLE_RTOL."""
    dtype = api.DTYPES[dtype]
    kernel = cases.make_kernel(case, Sampler(dtype, "cpu"))
    factors = api.separable_factors(kernel)
    assert factors is not None
    col, row = factors
    assert col.dtype == row.dtype == torch.float32
    assert col.shape == (case["kh"],) and row.shape == (case["kw"],)
    assert (col >= 0).all() and (row >= 0).all()
    k = kernel.double()
    rel = torch.linalg.matrix_norm(residual(kernel, col, row)) / torch.linalg.matrix_norm(k)
    assert rel <= api.SEPARABLE_RTOL[dtype]


@pytest.mark.parametrize("dtype", api.DTYPES)
def test_gaussian_sampled_from_sigma_is_separable(dtype):
    """A Gaussian sampled directly as exp(-(x^2 + y^2) / 2 sigma^2), not built as an outer
    product, is rank-1 only up to rounding; it must still be accepted."""
    dtype = api.DTYPES[dtype]
    for k, sigma in [(3, 0.8), (7, 1.5), (21, 3.0), (31, 8.0)]:
        x = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma * sigma))
        assert api.separable_factors((g / g.sum()).to(dtype)) is not None, (k, sigma)


@pytest.mark.parametrize("dtype", api.DTYPES)
def test_nonseparable_kernels_are_rejected(dtype):
    """The source's random kernels and its example 2 kernel take the direct path, and so
    does a separable kernel perturbed by 8 u_d (twice the threshold) in a corner tap, a
    direction nearly orthogonal to its rank-1 term (a bump on the centre tap would mostly
    be absorbed by it)."""
    dtype = api.DTYPES[dtype]
    for case in RANDOM_CASES:
        kernel = cases.make_kernel(case, Sampler(dtype, "cpu", seed=2))
        if case["kh"] * case["kw"] > 1:
            assert api.separable_factors(kernel) is None, case_id(case)
    example2 = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])
    assert api.separable_factors(example2.to(dtype)) is None
    g = cases.gaussian_2d(7, 7)
    g[0, 6] += 8 * api.SEPARABLE_RTOL[dtype] / 4 * torch.linalg.matrix_norm(g)
    assert api.separable_factors(g.to(dtype)) is None


def test_degenerate_kernels_are_separable():
    """1 x 1, 1 x KW, KH x 1, the identity, a box and an all-zero kernel are rank <= 1."""
    identity = torch.zeros(3, 3)
    identity[1, 1] = 1
    for kernel in [
        torch.ones(1, 1),
        torch.rand(1, 7) + 0.1,
        torch.rand(5, 1) + 0.1,
        identity,
        torch.full((3, 3), 0.111111),
        torch.zeros(5, 3),
    ]:
        col, row = api.separable_factors(kernel)
        assert torch.allclose(torch.outer(col, row), kernel, rtol=4 * U32, atol=0)


def two_pass_bound(image, kernel, col, row, exact, dtype):
    """Bound for the two-pass result against the exact 2-D sum with the supplied kernel.
    Each product goes through a KW-term and a KH-term FP32 sum, so the summation term
    counts KH + KW, not KH * KW, and the discarded residual R = kernel - outer(col, row)
    adds sum |x| |R|:
    |y_hat - y| <= u_d |y| + (1 + u_d) (gamma_{KH+KW} sum |x| |col| |row| + sum |x| |R|) + eta.
    """
    (h, w), (kh, kw) = image.shape, kernel.shape
    win = windows(image, kh, kw).abs()
    factored = (win @ torch.outer(col.double(), row.double()).abs().reshape(-1)).view(h, w)
    dropped = (win @ residual(kernel, col, row).abs().reshape(-1)).view(h, w)
    info = torch.finfo(dtype)
    u, eta = info.eps / 2, info.smallest_normal * info.eps / 2
    return u * exact.abs() + (1 + u) * (gamma(kh + kw) * factored + dropped) + eta


def contract_bound(image, kernel):
    """(exact, bound) for any result the contract allows: the direct FP32-accumulation
    bound with n = KH * KW terms, widened to the two-pass bound when separable_factors
    accepts the kernel. Both scale with the kernel, so unlike the fixed TOLERANCES they
    stay tight for small kernels and grow for large ones."""
    exact, magnitude = oracle(image, kernel)
    (kh, kw), dtype = kernel.shape, image.dtype
    bound = error_bound(exact, magnitude, kh * kw, dtype)
    factors = api.separable_factors(kernel)
    if factors is not None:
        bound = torch.maximum(bound, two_pass_bound(image, kernel, *factors, exact, dtype))
    return exact, bound


def bound_ratio(out, exact, bound):
    return ((out.double().cpu() - exact).abs() / bound).max().item()


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", GAUSSIAN_CASES, ids=case_id)
def test_two_pass_matches_oracle(case, dtype):
    """The two-pass algorithm itself stays within two_pass_bound in every dtype."""
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    image, kernel = cases.make_inputs(case, Sampler(dtype, device(), seed=3))
    col, row = api.separable_factors(kernel)
    exact, _ = oracle(image, kernel)
    bound = two_pass_bound(image, kernel, col, row, exact, dtype)
    worst = bound_ratio(two_pass(image, col, row), exact, bound)
    assert worst <= 1, f"two-pass error exceeds its bound by {worst:.2f}x"


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
@pytest.mark.parametrize("backend", [b for b in api.BACKENDS if b != "pytorch"])
def test_matches_oracle(backend, case, dtype):
    """Every backend against the exact result within contract_bound, the check that
    scales with the kernel. With normalized blur kernels the fixed FP16/BF16 tolerances
    cannot see reduced-precision accumulation at small kernels; this bound can (see
    test_bound_rejects_reduced_precision)."""
    skip_if_unavailable(backend, api)
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    image, kernel = cases.make_inputs(case, Sampler(dtype, device(), seed=8))
    out = call_or_skip(api, backend, (image, kernel))
    worst = bound_ratio(out, *contract_bound(image, kernel))
    assert worst <= 1, f"error exceeds the contract bound by {worst:.2f}x"


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", GAUSSIAN_CASES + [cases.BENCH_CASES[2]], ids=case_id)
def test_two_pass_within_tolerance(case, dtype):
    """Factoring must not move results outside the fixed TOLERANCES against the direct
    reference, on every Gaussian test case and the 4096 x 4096, 21 x 21 bench case."""
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    if case["h"] * case["w"] > 1 << 20 and device() == "cpu":
        pytest.skip("bench-size case runs on the GPU only")
    image, kernel = cases.make_inputs(case, Sampler(dtype, device(), seed=4))
    col, row = api.separable_factors(kernel)
    rtol, atol = api.TOLERANCES[dtype]
    torch.testing.assert_close(
        two_pass(image, col, row), api.gaussian_blur(image, kernel), rtol=rtol, atol=atol
    )


# --- Tolerances -----------------------------------------------------------------------


def direct(image, kernel, *, order=1, tf32=False, accumulate=torch.float32):
    """Direct 2-D filter as one running sum over the taps in row-major order (order=1) or
    reverse (order=-1), in a chosen accumulation precision, rounded once to image.dtype."""
    (h, w), (kh, kw) = image.shape, kernel.shape
    x = F.pad(image.float(), (kw // 2, kw // 2, kh // 2, kh // 2))
    k = kernel.float()
    if tf32:  # round to nearest, 10 explicit mantissa bits (13 FP32 bits dropped)
        x, k = (
            (t.contiguous().view(torch.int32) + 0x1000).bitwise_and(~0x1FFF).view(torch.float32)
            for t in (x, k)
        )
    acc = torch.zeros(h, w, dtype=accumulate, device=image.device)
    for t in range(kh * kw)[::order]:
        m, n = divmod(t, kw)
        acc = (acc.float() + x[m : m + h, n : n + w] * k[m, n]).to(accumulate)
    return acc.to(image.dtype)


TOLERANCE_CASES = [
    cases._case(96, 131, k, k, kind) for k in (21, 31) for kind in ("gaussian", "random")
]


@pytest.mark.parametrize("order", [1, -1], ids=["row-major", "reverse"])
@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("case", TOLERANCE_CASES, ids=case_id)
def test_tolerances_cover_summation_order(case, dtype, order):
    """Any correct backend differs from the reference by summation order and one final
    rounding. A single running FP32 sum over all KH * KW taps, the longest chain any
    backend would use, must pass the fixed TOLERANCES at the largest kernels."""
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    image, kernel = cases.make_inputs(case, Sampler(dtype, device(), seed=5))
    rtol, atol = api.TOLERANCES[dtype]
    expected = api.gaussian_blur(image, kernel)
    torch.testing.assert_close(direct(image, kernel, order=order), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "dtype, mode",
    [
        ("fp32", {"tf32": True}),
        ("fp16", {"accumulate": torch.float16}),
        ("bf16", {"accumulate": torch.bfloat16}),
    ],
    ids=["fp32-tf32-products", "fp16-fp16-accumulation", "bf16-bf16-accumulation"],
)
def test_tolerances_reject_reduced_precision(dtype, mode):
    """Negative controls at the source's largest kernel, 21 x 21: the tolerances must catch
    TF32 products and reduced-precision accumulation. The image is the source's
    performance data, [0, 255), where partial sums grow large enough to lose bits."""
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    s = Sampler(dtype, device(), seed=6)
    image = s.uniform(0, 255, 96, 131)
    kernel = cases.gaussian_2d(21, 21).to(dtype).to(image.device)
    rtol, atol = api.TOLERANCES[dtype]
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            direct(image, kernel, **mode), api.gaussian_blur(image, kernel), rtol=rtol, atol=atol
        )


@pytest.mark.parametrize(
    "dtype, mode",
    [
        ("fp32", {"tf32": True}),
        ("fp16", {"accumulate": torch.float16}),
        ("bf16", {"accumulate": torch.bfloat16}),
    ],
    ids=["fp32-tf32-products", "fp16-fp16-accumulation", "bf16-bf16-accumulation"],
)
def test_bound_rejects_reduced_precision(dtype, mode):
    """The same shortcuts at the source's performance kernel size, 7 x 7, on unit-normal
    data, where the fixed FP16/BF16 tolerances pass them (ratios 0.12-0.16, docs/design.md):
    contract_bound must still reject them."""
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    image, kernel = cases.make_inputs(cases._case(64, 67, 7, 7), Sampler(dtype, device(), seed=9))
    assert bound_ratio(direct(image, kernel, **mode), *contract_bound(image, kernel)) > 1


# --- Contract -------------------------------------------------------------------------


def test_source_examples():
    """Both worked examples from the source statement."""
    image = torch.arange(1.0, 26.0).view(5, 5)
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    expected = torch.tensor(
        [
            [1.6875, 2.75, 3.5, 4.25, 3.5625],
            [4.75, 7.0, 8.0, 9.0, 7.25],
            [8.5, 12.0, 13.0, 14.0, 11.0],
            [12.25, 17.0, 18.0, 19.0, 14.75],
            [11.0625, 15.25, 16.0, 16.75, 12.9375],
        ]
    )
    # Dyadic kernel and integer image: every partial sum is exact, so equality is exact.
    assert torch.equal(api.gaussian_blur(image, kernel), expected)
    image = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]])
    kernel = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])
    expected = torch.tensor([[13.0, 23.0, 19.0], [31.0, 50.0, 39.0], [31.0, 47.0, 37.0]])
    torch.testing.assert_close(api.gaussian_blur(image, kernel), expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("dtype", api.DTYPES.values())
@pytest.mark.parametrize("case", cases.TEST_CASES, ids=case_id)
def test_reference_contract(case, dtype):
    """Small integers: every partial sum is exact in FP32, so an FP32-accumulating result
    is the exact sum rounded once to `dtype`. FP16 or BF16 accumulation would round partial
    sums above 2048 or 256 and fail. Both tensors start one element into their storage."""
    h, w, kh, kw = case["h"], case["w"], case["kh"], case["kw"]
    image = (torch.arange(h * w + 1) % 7 - 3).to(dtype)[1:].view(h, w)
    kernel = (torch.arange(kh * kw + 1) % 5 - 1).to(dtype)[1:].view(kh, kw)
    assert image.storage_offset() == 1 and kernel.storage_offset() == 1
    image_before, kernel_before = image.clone(), kernel.clone()
    out = api.gaussian_blur(image, kernel)
    exact, _ = oracle(image, kernel)
    assert out.shape == (h, w) and out.dtype == dtype and out.device == image.device
    assert out.is_contiguous() and out.storage_offset() == 0
    assert torch.equal(out, exact.to(dtype))
    out.fill_(-1)  # a fresh buffer: writing it cannot reach the inputs
    assert torch.equal(image, image_before) and torch.equal(kernel, kernel_before)


def test_reference_is_correlation_not_convolution():
    """An asymmetric kernel pins orientation and the padding origin: out[i, j] reads
    image[i + m - KH//2, j + n - KW//2] * kernel[m, n]."""
    image = torch.zeros(5, 5)
    image[2, 2] = 1
    kernel = torch.arange(1.0, 16.0).view(3, 5)
    out = api.gaussian_blur(image, kernel)
    # An impulse at (2, 2) reaches out[2 - (m - 1), 2 - (n - 2)] = kernel[m, n]:
    # the kernel appears rotated by 180 degrees around the impulse.
    expected = torch.zeros(5, 5)
    expected[1:4, 0:5] = kernel.flip(0, 1)
    assert torch.equal(out, expected)


def test_zero_padding_not_clamp():
    """A constant image keeps its value in the interior but loses the padded taps at the
    borders: with a 3 x 3 box, corners see 4/9 and edges 6/9 of the kernel."""
    out = api.gaussian_blur(torch.ones(4, 6), torch.full((3, 3), 1 / 9))
    assert out[1:-1, 1:-1].sub(1).abs().max() < 1e-6
    torch.testing.assert_close(out[0, 0], torch.tensor(4 / 9))
    torch.testing.assert_close(out[0, 2], torch.tensor(6 / 9))


@pytest.mark.parametrize("which", ["image", "kernel", "both"])
@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("backend", api.BACKENDS)
def test_misaligned_input(backend, dtype, which):
    """Inputs starting one element into their storage are not 16-byte aligned, and odd W
    misaligns every row after the first, so any vectorized or async-copy path must fall
    back. Compared with the reference on aligned copies of the same values."""
    skip_if_unavailable(backend, api)
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    h, w, kh, kw = 45, 131, 7, 9
    s = Sampler(dtype, device(), seed=7)
    image_full = s.randn(h * w + 1)
    kernel_full = cases.make_kernel(cases._case(1, 1, kh, kw), s).reshape(-1)
    kernel_full = torch.cat([kernel_full[:1], kernel_full])
    image = (image_full[1:] if which in ("image", "both") else image_full[:-1]).view(h, w)
    kernel = (kernel_full[1:] if which in ("kernel", "both") else kernel_full[:-1]).view(kh, kw)
    for name, t in (("image", image), ("kernel", kernel)):
        if which in (name, "both"):
            assert t.data_ptr() % 16 != 0
    out = call_or_skip(api, backend, (image, kernel))
    expected = api.gaussian_blur(image.clone(), kernel.clone())
    rtol, atol = api.TOLERANCES[dtype]
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


def test_rejects_invalid_inputs():
    image, kernel = torch.zeros(8, 9), torch.zeros(3, 5)
    bad = [
        (image.view(-1), kernel),  # 1-D
        (image[None], kernel),  # 3-D
        (image, kernel.view(-1)),
        (image[:, ::2], kernel),  # non-contiguous
        (image, torch.zeros(3, 10)[:, ::2]),
        (image, torch.zeros(2, 3)),  # even KH
        (image, torch.zeros(3, 4)),  # even KW
        (image, torch.zeros(0, 3)),  # empty kernel
        (torch.zeros(0, 9), kernel),  # empty image
        (image, kernel.half()),  # dtype mismatch
        (image.double(), kernel.double()),  # unsupported dtypes
        (image.int(), kernel.int()),
        (image.to_sparse(), kernel),
        (torch.zeros(8, 9, requires_grad=True), kernel),
    ]
    for args in bad:
        with pytest.raises(ValueError):
            api.gaussian_blur(*args)
    with pytest.raises(TypeError):
        api.gaussian_blur([[1.0]], kernel)
    if torch.cuda.is_available():
        with pytest.raises(ValueError):
            api.gaussian_blur(torch.zeros(8, 9, device="cuda"), kernel)


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_gpu_backends_reject_cpu(backend):
    with pytest.raises(ValueError, match="requires NVIDIA CUDA"):
        api.gaussian_blur(torch.zeros(8, 9), torch.zeros(3, 3), backend=backend)


def test_traffic_model():
    c = cases._case(4096, 4096, 21, 21)
    # Image and output once each plus the kernel; FLOPs count the direct filter.
    assert cases.logical_bytes(c, torch.float32) == 4 * (2 * 4096 * 4096 + 441)
    assert cases.logical_bytes(c, torch.float16) == 2 * (2 * 4096 * 4096 + 441)
    assert cases.flops(c) == 2 * 4096 * 4096 * 441


# --- CUDA variants --------------------------------------------------------------------


def cuda_extension():
    skip_if_unavailable("cuda")
    from problems.gaussian_blur.cuda.implementation import extension

    return extension()


def test_cuda_alias_is_direct():
    assert dispatch.resolve(api, "cuda") == ("cuda", {})
    assert api.VARIANTS["cuda_direct"] == ("cuda", {"variant": 0})
    assert api.VARIANTS["cuda_direct_const"] == ("cuda", {"variant": 1})


def test_cuda_const_capacity():
    """127 x 129 = 16383 taps fit the 64 KiB __constant__ buffer; 129 x 129 = 16641 do not,
    through the API (ValueError) and the binding (RuntimeError). The kernel is far larger
    than the image, so most taps read padding; checked against the K-scaled contract bound,
    since 16383 FP32 terms outgrow the fixed tolerance."""
    from problems.gaussian_blur.cuda.implementation import MAX_CONST_TAPS

    ext = cuda_extension()
    assert ext.max_const_taps() == MAX_CONST_TAPS == 16384
    s = Sampler(torch.float32, "cuda", seed=13)
    image = s.randn(9, 11)
    kernel = s.uniform(0.0001, 0.02, 127, 129)
    out = api.gaussian_blur(image, kernel, backend="cuda_direct_const")
    worst = bound_ratio(out, *contract_bound(image, kernel))
    assert worst <= 1, f"error exceeds the contract bound by {worst:.2f}x"
    big = s.uniform(0.0001, 0.02, 129, 129)
    with pytest.raises(ValueError, match="16384"):
        api.gaussian_blur(image, big, backend="cuda_direct_const")
    with pytest.raises(RuntimeError, match="16384"):
        ext.gaussian_blur(image, big, torch.empty_like(image), 1)


def test_cuda_const_reloads_kernel():
    """__constant__ memory persists between calls: each call must upload its own kernel,
    including a smaller one after a larger one and a different shape with the same tap
    count, with no host synchronization in between."""
    skip_if_unavailable("cuda")
    s = Sampler(torch.float32, "cuda", seed=14)
    image = s.randn(67, 131)
    kernels = [
        s.uniform(0.0001, 0.02, 21, 21),
        cases.gaussian_2d(3, 3).float().cuda(),
        s.uniform(0.0001, 0.02, 21, 21),
        cases.gaussian_2d(7, 5).float().cuda(),
        cases.gaussian_2d(5, 7).float().cuda(),
    ]
    outs = [api.gaussian_blur(image, k, backend="cuda_direct_const") for k in kernels]
    rtol, atol = api.TOLERANCES[torch.float32]
    for kernel, out in zip(kernels, outs, strict=True):
        torch.testing.assert_close(out, api.gaussian_blur(image, kernel), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", api.DTYPES)
@pytest.mark.parametrize("shape", [(8, 32), (9, 33), (4100, 3), (3, 4100)], ids=str)
def test_cuda_grid_tails(shape, dtype):
    """32 x 8 thread blocks: exact multiples, one-past tails in both axes, and images far
    taller or wider than a block row, against the reference."""
    skip_if_unavailable("cuda")
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    h, w = shape
    image, kernel = cases.make_inputs(cases._case(h, w, 5, 7), Sampler(dtype, "cuda", seed=10))
    rtol, atol = api.TOLERANCES[dtype]
    for backend in api.VARIANTS:
        out = api.gaussian_blur(image, kernel, backend=backend)
        torch.testing.assert_close(out, api.gaussian_blur(image, kernel), rtol=rtol, atol=atol)


@pytest.mark.parametrize("variant", sorted({o["variant"] for _, o in api.VARIANTS.values()}))
@pytest.mark.parametrize("dtype", api.DTYPES)
def test_binding_misaligned_output(dtype, variant):
    """The API always allocates an aligned output; through the extension an output one
    element into its storage must work and leave its neighbours untouched."""
    ext = cuda_extension()
    dtype = api.DTYPES[dtype]
    skip_unsupported_dtype(dtype)
    h, w = 45, 131
    image, kernel = cases.make_inputs(cases._case(h, w, 7, 9), Sampler(dtype, "cuda", seed=12))
    storage = torch.full((h * w + 2,), -7.0, dtype=dtype, device="cuda")
    out = storage[1:-1].view(h, w)
    ext.gaussian_blur(image, kernel, out, variant)
    rtol, atol = api.TOLERANCES[dtype]
    torch.testing.assert_close(out, api.gaussian_blur(image, kernel), rtol=rtol, atol=atol)
    assert storage[0].item() == storage[-1].item() == -7.0


def test_binding_rejects_invalid_arguments():
    ext = cuda_extension()
    image, kernel = torch.zeros(8, 9, device="cuda"), torch.zeros(3, 5, device="cuda")
    out = torch.empty_like(image)
    bad = [
        (image, kernel, out, 2),  # variant
        (image, kernel, out, -1),
        (image, kernel, torch.empty(8, 8, device="cuda"), 0),  # output shape
        (image, kernel, out.half(), 0),  # output dtype
        (image, kernel.half(), out, 0),  # kernel dtype
        (image, torch.zeros(2, 3, device="cuda"), out, 0),  # even KH
        (image, torch.zeros(3, 4, device="cuda"), out, 0),  # even KW
        (image, kernel, torch.empty(8, 18, device="cuda")[:, ::2], 0),  # non-contiguous
        (image.view(-1), kernel, out.view(-1), 0),  # 1-D
        (image.double(), kernel.double(), out.double(), 0),
        (image.cpu(), kernel.cpu(), out.cpu(), 0),
        (image, kernel, image, 0),  # output aliases the image
    ]
    for args in bad:
        with pytest.raises(RuntimeError):
            ext.gaussian_blur(*args)


def test_cuda_rejects_invalid_variant():
    skip_if_unavailable("cuda")
    from problems.gaussian_blur.cuda.implementation import gaussian_blur

    image, kernel = torch.zeros(8, 9, device="cuda"), torch.zeros(3, 3, device="cuda")
    with pytest.raises(ValueError):
        gaussian_blur(image, kernel, variant=2)
