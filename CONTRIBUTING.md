# Contributing a study

Read the problem's source, prerequisites, and proposed experiments. Define the
contract, then follow the [implementation and testing methodology](docs/methodology.md),
[benchmarking plan](docs/benchmarking.md), and [profiling plan](docs/profiling.md).
Create backend directories only when real work starts. Keep baselines and name
variants for their mechanism; do not copy source statements or solutions.

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
roadmap table owns the numbering. A started study follows the
[Vector Addition](problems/vector_add/README.md) layout:

```text
<problem>/
├── README.md        overview: status, headline chart, key results, quick start, page map
├── api.py           validated public entry point
├── cuda/ triton/ pytorch/ jax/   one directory per backend actually implemented
├── tests/           correctness and benchmark-statistics tests
├── benchmarks/      run.py, report.py, plot.py, sweep.sh
├── figures/         generated charts (tracked)
├── docs/
│   ├── design.md        source, contract, variants, performance model
│   ├── results.md       environment, tables, profiler evidence, limitations
│   ├── testing.md       setup, correctness suite, sanitizer
│   ├── benchmarking.md  scopes, options, reproduction, output format
│   └── profiling.md     profiler commands and what they test
└── results/         raw JSON and profiler output (ignored by Git)
```

Keep the README short; detail belongs in `docs/`. Each docs page starts with the
same navigation line.

## Checks

CI runs `ruff`, `clang-format` and the CPU test suite on every push and pull
request. GPU backends skip in CI, so run the full suite, Compute Sanitizer and
benchmarks on NVIDIA hardware and record the outcome in `docs/results.md`.
