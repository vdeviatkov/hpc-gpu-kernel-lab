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

## Layout and checks

Problem directories use plain importable names (`problems/vector_add`); the
roadmap table owns the numbering. CI runs `ruff`, `clang-format` and the CPU
test suite on every push and pull request. GPU backends skip in CI, so run the
full suite, Compute Sanitizer and benchmarks on NVIDIA hardware and record the
results in the problem README.
