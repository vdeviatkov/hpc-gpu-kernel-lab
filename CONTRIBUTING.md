# Contributing a study

Read the problem's source, prerequisites, and proposed experiments. Define the
contract, then follow the [implementation and testing methodology](docs/methodology.md),
[benchmarking plan](docs/benchmarking.md), and [profiling plan](docs/profiling.md).
Every problem starts from a generated scaffold whose stubs raise
`NotImplementedError`. Keep baselines and name variants for their mechanism; do
not copy source statements or solutions.

## Status

Maintain status in the problem README and the [main roadmap](README.md#roadmap).
There is no separate progress dashboard or coverage counter to synchronize.

- **Planned:** a study plan, with no accepted implementation.
- **In Progress:** implementation or validation is underway.
- **Complete:** planned backend scope and correctness are satisfied, with real
  GPU measurements, reproducible commands/environment, and limitations documented.
- **Optimized:** a named change additionally has benchmark/profiler evidence and
  an explanation of its wins and regressions.

If backend scope changes, explain why and update the roadmap cells. Put real
results with the problem; retain raw samples and use generated reports as views.
Until measured, write **Results pending hardware benchmark**.

## Problem layout

Problem directories use plain importable names (`problems/vector_add`); the
roadmap table owns the numbering. Shared code lives in `lab/` (dispatch, input
sampling, test helpers, benchmark harness and report). Shared CUDA headers live in
`lab/cuda/include/lab/` and are on every extension's include path:

- `lab/vector.cuh`: 16-byte `Pack<T>`, `load_pack`/`store_pack`, host `pack_aligned`;
- `lab/launch.cuh`: `global_index`, `grid_threads`, `grid_blocks`;
- `lab/checks.h`: tensor checks for `bindings.cpp`.

Share mechanics only; each kernel body stays in its problem. A completed study follows
the [Vector Addition](problems/vector_add/README.md) layout:

```text
<problem>/
├── README.md        overview: status, headline chart, key results, quick start, page map
├── api.py           public entry point: contract, BACKENDS, DTYPES, TOLERANCES
├── cases.py         test/benchmark inputs, logical bytes and FLOPs
├── pytorch/ cuda/ triton/ jax/   one directory per planned backend
├── tests/           correctness tests against the PyTorch reference
├── benchmarks/      optional: problem-specific runners, plots, sweeps
├── figures/         generated charts (tracked)
└── docs/
    ├── design.md        source, contract, variants, performance model
    ├── results.md       environment, tables, profiler evidence, limitations
    ├── testing.md       setup, correctness suite, sanitizer
    ├── benchmarking.md  scopes, options, reproduction, output format
    └── profiling.md     profiler commands and what they test
```

Keep the README short; detail belongs in `docs/`, `figures/` and `benchmarks/`,
which are added when a study is measured. Each docs page starts with the same
navigation line.

## Checks

CI runs `ruff`, `clang-format` and the CPU test suite on every push and pull
request. GPU backends skip in CI, so run the full suite, Compute Sanitizer and
benchmarks on NVIDIA hardware and record the outcome in `docs/results.md`.
