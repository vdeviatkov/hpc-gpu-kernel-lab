# Testing

[← Vector Addition](../README.md) · [Design](design.md) · [Results](results.md) ·
**Testing** · [Benchmarking](benchmarking.md) · [Profiling](profiling.md)

## Setup

From the repository root, use Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r problems/vector_add/requirements.txt
```

On Linux NVIDIA hosts, use a CUDA-enabled PyTorch wheel, matching driver/toolkit
and supported C++ compiler. Make nvcc discoverable through `CUDA_HOME` or `PATH`.
Install Triton separately if PyTorch does not supply it. JAX is optional; install
the CUDA extra appropriate to your setup. See the
[PyTorch selector](https://pytorch.org/get-started/locally/) and
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

The [loader](../cuda/implementation.py) builds the extension lazily with Ninja via
[PyTorch's extension API](https://docs.pytorch.org/docs/stable/cpp_extension.html).
It inherits PyTorch's required C++ standard and ABI, uses O3/line information,
and does not enable fast math. Set `TORCH_EXTENSIONS_DIR=artifacts/torch_extensions`
to keep the build cache in the repository, and record any explicit
`TORCH_CUDA_ARCH_LIST` (the recorded results used `12.0`).

## Correctness suite

```bash
python -m pytest -q problems/vector_add
ruff check problems
ruff format --check problems
clang-format --dry-run --Werror problems/vector_add/cuda/*.{cpp,cu}
```

Every backend runs through the same parametrized tests
([test_vector_add.py](../tests/test_vector_add.py)):

- Empty, odd and large vectors; partial-pack tails; storage offsets 0, 1 and 3.
- Output alignment independent of the inputs, for every dtype, with guard values
  around the output that must stay untouched.
- Input preservation, aliased inputs, disjoint slices of one storage.
- Signed zeros, cancellation, ±Inf and NaN.
- Multi-iteration grid-stride loops for every dtype (more than 4096 × 256 packs).
- Launch configurations, nondefault streams, and rejection of invalid contracts.

The reference adds FP32-converted inputs before rounding to the output dtype.
Tolerances are explicit in [api.py](../api.py); guard/ownership checks are exact.
Missing hardware or dependencies are explicit skips; build and correctness failures
fail. Inspect skips on the target host before accepting results.
[test_benchmark.py](../tests/test_benchmark.py) covers the percentile statistics
and the timer's exclusion of initialization.

## Compute Sanitizer

```bash
compute-sanitizer --tool memcheck python -m pytest -q problems/vector_add/tests/test_vector_add.py \
  -k 'cuda and (values_and_guards or alignment or second_iteration or stream)'
```

## Continuous integration

[CI](../../../.github/workflows/ci.yml) runs lint, formatting and the CPU suite on
every push and pull request. GPU backends skip there, so the full suite and the
sanitizer run on NVIDIA hardware. The latest outcome is recorded in
[results](results.md#validation).
