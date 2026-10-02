# Optimization methodology

The unit of progress is an explained experiment. A source challenge supplies a
workload, but a useful portfolio study must connect an algorithm, a hardware
model, and measurements. Nothing in this document is an implemented framework.

## Workflow

1. **Understand the algorithm.** State shapes, layouts, mutation, dtype,
   accumulation, output precision, and boundary behavior.
2. **Build a correctness reference.** Match the source contract first; label
   local extensions and numerical approximations separately.
3. **Derive a simple performance model.** Estimate useful work, traffic,
   temporary storage, parallelism, and a plausible lower bound.
4. **Implement a straightforward GPU version.** Make ownership and memory
   access visible before introducing sophisticated scheduling.
5. **Benchmark.** Follow the [timing and environment plan](benchmarking.md).
6. **Profile.** Follow the [profiling plan](profiling.md) to test a specific hypothesis.
7. **Identify the bottleneck.** Distinguish bandwidth, instruction/dependency
   throughput, synchronization, resource limits, and launch overhead.
8. **Form an optimization hypothesis.** Predict what should change in both
   latency and hardware evidence.
9. **Implement the optimization.** Preserve the baseline and give the variant
   a meaningful mechanism-based name.
10. **Measure again.** Repeat correctness, benchmark the same cases, and profile
    the expected hardware effect. Include regressions and crossover sizes.
11. **Explain why performance changed.** Compare the prediction to observations,
    discuss alternatives, and define the next experiment.

## Questions to answer

- Is the workload memory-bound, compute-bound, synchronization-bound, or launch-bound at this shape?
- How many bytes must move, and through which memory level?
- How many useful FLOPs or integer operations are required?
- What is a plausible theoretical lower bound, and what costs does it omit?
- Are loads and stores coalesced across neighboring lanes?
- Is shared memory providing reuse or only adding traffic and barriers?
- Are shared-memory bank conflicts present?
- Does occupancy limit latency hiding, or would extra registers improve useful work?
- Are synchronization or dependency chains preventing issue?
- Are registers spilling into local memory?
- Is launch overhead dominant for small inputs or multi-pass algorithms?
- Is fusion useful after accounting for resources and lost parallelism?
- Are we trading recomputation for reduced memory traffic?

## Correctness and tests

Use pytest for Python interfaces. Keep problem-specific cases in that problem's
`tests/` directory; extract shared checks only when multiple studies need them.
Before timing, compare every implementation with an independent reference.

Cover small/large and non-power-of-two shapes, tails, supported strides and
alignment, numerical extremes, and exact integer behavior. Set dtype-specific
error budgets and inspect absolute error near cancellation. For contention,
sparsity, and routing, vary distributions as well as shapes.

Check mutation/aliasing, output ownership, current streams, and device placement.
Use GPU memory/race/synchronization checks for relevant low-level changes.
Missing hardware is a skip; compilation or correctness failures are failures.

## Repository organization

Create `cuda/`, `triton/`, `pytorch/`, or `jax/` inside a problem when its first
implementation exists. Add local `tests/`, `benchmarks/`, `docs/` and `figures/` as needed.
Avoid empty source files and placeholder-only infrastructure directories.

Extract common timing, correctness, environment, and CUDA helpers only after
concrete reuse appears. Keep launches, allocations, ownership, and synchronization
visible. Add shared scripts/build/CI only for actual executable work.

## Performance models

For an operation with useful work `F`, logical traffic `B`, compute ceiling `P`,
and relevant bandwidth `W`, a first lower bound is `max(F/P, B/W)`. It omits
launches, dependencies, barriers, padding, and many occupancy effects. Arithmetic
intensity is `F/B`; a roofline comparison must specify the memory level and
precision of its ceilings.

Vector addition provides a minimum-traffic model of `3 * N * bytes_per_element`.
Dense matrix multiplication has approximately `2*M*N*K` useful FLOPs, but its
actual traffic depends on reuse. Reductions add partial-buffer traffic; online
attention changes intermediate storage. Never equate a lower-bound traffic
estimate to measured DRAM bytes. For exponentials, integer hashing, or sparse
routing, do not invent an arbitrary FLOP count merely to populate a chart.

## Naming experiments

Prefer names such as `scalar_coalesced`, `aligned_vector_loads`, `shared_tree`,
`warp_shuffle`, `padded_shared_tile`, `register_tile`, or `online_softmax`.
Keep numerical changes and precision modes in the name or configuration. A
variant called optimized is not proof that it is faster.

## Evidence note template

For each optimization, record the workload case, expected bottleneck, exact
change, predicted counter behavior, measured latency distribution, observed
counter changes, numerical effects, and an alternative explanation. Finish
with one experiment that could disprove the explanation.

Retain negative results. More occupancy, fewer instructions, or
higher cache hit rate is useful only if it explains better useful execution.
Do not introduce Tensor Cores, asynchronous copies, or persistence merely to
make a source file look advanced.
