"""Benchmark harness shared by every problem.

Generic runner, from the repository root:

    python -m lab.bench relu --output artifacts/relu.json
    python -m lab.report artifacts/relu.json

A problem provides `api.py` (BACKENDS, DTYPES, TOLERANCES, FUNCTION) and `cases.py`
(BENCH_CASES, make_inputs, logical_bytes, flops). Problems with special timing needs
(such as vector_add) build their own runner from the pieces below.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import torch

from lab import dispatch
from lab.data import Sampler, clone_tensors

SCHEMA_VERSION = 1


def summarize(samples):
    if not samples or any(x <= 0 for x in samples):
        raise ValueError("need positive latency samples")
    ordered = sorted(samples)

    def percentile(q):
        index = (len(ordered) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {
        "p20": percentile(0.2),
        "median": percentile(0.5),
        "p50": percentile(0.5),
        "p80": percentile(0.8),
        "mean": statistics.mean(samples),
        "stddev": statistics.pstdev(samples),
    }


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True, timeout=15)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(device):
    props = torch.cuda.get_device_properties(device)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "os": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "cuda_runtime": torch.version.cuda,
        "cuda_toolkit": command("nvcc", "--version"),
        "nvidia_smi": command(
            "nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"
        ),
        "gpu": props.name,
        "device": device,
        "uuid": str(props.uuid) if hasattr(props, "uuid") else None,
        "compute_capability": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "memory_bytes": props.total_memory,
        "l2_bytes": getattr(props, "L2_cache_size", None),
        "commit": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--porcelain"),
        "flags": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_LAUNCH_BLOCKING",
                "TORCH_CUDA_ARCH_LIST",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_FLAGS",
            )
        },
    }


def measure(fn, sync, scope, warmup, samples):
    result = fn()  # Build/JIT and allocator initialization are never timed.
    sync(result)
    for _ in range(warmup):
        result = fn()
    sync(result)
    start = end = None
    if scope == "device":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        end.synchronize()  # Materialize events before the first measured call.
    times = []
    for _ in range(samples):
        if scope == "device":
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000)
        else:
            before = time.perf_counter_ns()
            result = fn()
            sync(result)
            times.append((time.perf_counter_ns() - before) / 1000)
    return times


def _matching(rows, row, implementation):
    return next(
        (
            r
            for r in rows
            if r.get("implementation") == implementation
            and r.get("status") == "ok"
            and r.get("case", r.get("shape")) == row.get("case", row.get("shape"))
            and r["dtype"] == row["dtype"]
        ),
        None,
    )


def add_relative_metrics(rows, baseline="pytorch", reference=None):
    """Add speedup_vs_<baseline> and, given a reference row name, fraction_of_<reference>."""
    for row in rows:
        if row.get("status") != "ok" or row["implementation"] == reference:
            continue
        base = _matching(rows, row, baseline)
        if base and row is not base:
            row[f"speedup_vs_{baseline}"] = (
                base["latency_us"]["median"] / row["latency_us"]["median"]
            )
        ref = _matching(rows, row, reference) if reference else None
        if ref:
            row["fraction_of_copy"] = row["effective_gb_s"] / ref["effective_gb_s"]


def write_results(path, metadata, settings, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "environment": metadata,
        "settings": settings,
        "results": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def case_label(case):
    return ",".join(f"{k}={v}" for k, v in case.items())


COPY_REFERENCE = "copy_reference"


def time_copy_reference(row, inputs, args, sync):
    """Time a device-to-device copy of the first tensor input (one read, one write stream).

    A practical bandwidth target for problems that move about as many bytes as a copy.
    Like the backends, it includes allocation of the destination.
    """
    source = next(a for a in inputs if isinstance(a, torch.Tensor))
    values = measure(source.clone, sync, args.scope, args.warmup, args.samples)
    latency = summarize(values)
    nbytes = 2 * source.numel() * source.element_size()
    row.update(
        status="ok",
        raw_latency_us=values,
        latency_us=latency,
        logical_bytes=nbytes,
        effective_gb_s=nbytes / latency["median"] / 1000,
    )
    if args.peak_gb_s:
        row["fraction_of_peak"] = row["effective_gb_s"] / args.peak_gb_s
    print(
        f"{COPY_REFERENCE:22} {row['dtype']} {case_label(row['case']):28} "
        f"{latency['median']:.3f} us ({args.scope})"
    )
    return row


def run_problem(args):
    """Correctness-gated timing of every requested backend on every benchmark case."""
    api = importlib.import_module(f"problems.{args.problem}.api")
    cases = importlib.import_module(f"problems.{args.problem}.cases")
    function = api.FUNCTION
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("NVIDIA CUDA GPU required; no CPU performance fallback")
    torch.cuda.set_device(args.device)
    metadata = environment(args.device)
    backends = args.backends or list(api.BACKENDS)
    dtypes = args.dtypes or list(api.DTYPES)
    rng = random.Random(args.seed)
    rows = []

    def sync(_):
        torch.cuda.synchronize(args.device)

    for dtype_name in dtypes:
        dtype = api.DTYPES[dtype_name]
        for case in cases.BENCH_CASES:
            inputs = cases.make_inputs(case, Sampler(dtype, "cuda", args.seed))
            base = {"kernel": args.problem, "case": case, "dtype": dtype_name}
            try:
                expected = dispatch.call(api, function, "pytorch", *clone_tensors(inputs))
            except NotImplementedError as error:
                rows += [
                    {**base, "implementation": b, "status": "no_reference", "reason": str(error)}
                    for b in backends
                ]
                print(f"{args.problem} {dtype_name} {case_label(case)}: no PyTorch reference")
                continue
            order = list(backends)
            if args.copy_reference:
                order.append(COPY_REFERENCE)
            rng.shuffle(order)
            for backend in order:
                row = {**base, "implementation": backend, "scope": args.scope}
                if backend == COPY_REFERENCE:
                    rows.append(time_copy_reference(row, inputs, args, sync))
                    continue
                if backend == "jax" and args.scope == "device":
                    reason = "JAX runs on its own stream; use --scope api"
                    rows.append({**row, "status": "skipped", "reason": reason})
                    continue
                fn = partial(dispatch.call, api, function, backend, *clone_tensors(inputs))
                try:
                    actual = fn()
                except NotImplementedError as error:
                    rows.append({**row, "status": "not_implemented", "reason": str(error)})
                    continue
                rtol, atol = api.TOLERANCES[dtype]
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
                values = measure(fn, sync, args.scope, args.warmup, args.samples)
                latency = summarize(values)
                row.update(
                    status="ok",
                    correctness={"passed": True, "rtol": rtol, "atol": atol},
                    raw_latency_us=values,
                    latency_us=latency,
                )
                nbytes = cases.logical_bytes(case, dtype)
                flops = cases.flops(case)
                row["logical_bytes"] = nbytes
                row["effective_gb_s"] = nbytes / latency["median"] / 1000
                if flops:
                    row["flops"] = flops
                    row["tflop_s"] = flops / latency["median"] / 1e6
                if args.peak_gb_s:
                    row["fraction_of_peak"] = row["effective_gb_s"] / args.peak_gb_s
                rows.append(row)
                print(
                    f"{backend:22} {dtype_name} {case_label(case):28} "
                    f"{latency['median']:.3f} us ({args.scope})"
                )
    add_relative_metrics(rows, reference=COPY_REFERENCE if args.copy_reference else None)
    settings = {**vars(args), "output": str(args.output)}
    settings.update(
        cache_policy="reused inputs; no flushing",
        allocation_policy="output allocation included",
        backends=backends,
        dtypes=dtypes,
    )
    write_results(args.output, metadata, settings, rows)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("problem", help="directory name under problems/, e.g. relu")
    p.add_argument("--backends", nargs="+", help="default: all backends in api.BACKENDS")
    p.add_argument("--dtypes", nargs="+", help="default: all dtypes in api.DTYPES")
    p.add_argument("--scope", choices=["device", "api"], default="device")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--peak-gb-s", type=float, default=None, help="sourced DRAM bandwidth")
    p.add_argument("--notes", default="", help="clocks, power, thermal state, other GPU users")
    p.add_argument(
        "--copy-reference",
        action="store_true",
        help="also time a device copy of the first input and report %% of copy",
    )
    p.add_argument("--output", type=Path, help="default: artifacts/<problem>.json")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.warmup < 1 or args.samples < 1:
        p.error("warmup/samples must be positive")
    args.output = args.output or Path(f"artifacts/{args.problem}.json")
    run_problem(args)


if __name__ == "__main__":
    main()
