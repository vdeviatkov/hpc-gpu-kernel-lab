"""Run from the repository root: python -m problems.vector_add.benchmarks.run."""

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

from problems.vector_add import api

# Device-to-device copy (one read and one write stream), measured per case as an
# achievable-bandwidth reference. Its read/write mix differs from addition (1:1 vs 2:1),
# so it is not a strict upper bound; --peak-gb-s records the sourced hardware ceiling.
REFERENCE = "copy_reference"


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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", nargs="+", type=int, default=[1, 1025, 1048576, 25000000])
    p.add_argument("--dtypes", nargs="+", choices=api.DTYPES, default=["fp32", "fp16", "bf16"])
    p.add_argument(
        "--backends", nargs="+", choices=(*api.BACKENDS, "jax"), default=list(api.BACKENDS)
    )
    p.add_argument("--scope", choices=["device", "api"], default="device")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--offset", type=int, default=0, help="element offset for all torch buffers")
    p.add_argument("--threads", type=int, choices=[128, 256, 512], default=256)
    p.add_argument("--block-size", type=int, choices=[256, 512, 1024, 2048], default=1024)
    p.add_argument("--num-warps", type=int, choices=[4, 8], default=4)
    p.add_argument("--output", type=Path, default=Path("artifacts/vector_add.json"))
    p.add_argument("--notes", default="", help="clocks, power, thermal state, other GPU users")
    p.add_argument(
        "--peak-gb-s",
        type=float,
        default=None,
        help="sourced theoretical DRAM bandwidth; recorded and used for fraction_of_peak",
    )
    p.add_argument("--no-reference", action="store_true", help=f"skip the {REFERENCE} reference")
    p.add_argument("--profile", action="store_true", help="one backend/shape/dtype, capture only")
    return p


def run(args):
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("NVIDIA CUDA GPU required; no CPU performance fallback")
    torch.cuda.set_device(args.device)
    if "jax" in args.backends:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    metadata = environment(args.device)
    rows = []
    rng = random.Random(args.seed)
    for dtype_name in args.dtypes:
        for n in args.sizes:
            if dtype_name == "bf16" and torch.cuda.get_device_capability(args.device)[0] < 8:
                rows.append(
                    {
                        "dtype": dtype_name,
                        "shape": [n],
                        "status": "skipped",
                        "reason": "benchmark policy requires SM80+ for BF16",
                    }
                )
                continue
            dtype = api.DTYPES[dtype_name]
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            # All backends receive identical representable values; transfers precede timing.
            a_cpu = torch.randn(n, generator=generator).to(dtype)
            b_cpu = torch.randn(n, generator=generator).to(dtype)
            a = torch.empty(n + args.offset, device="cuda", dtype=dtype)[args.offset :]
            b = (
                torch.empty_like(a)
                if not args.offset
                else torch.empty(n + args.offset, device="cuda", dtype=dtype)[args.offset :]
            )
            a.copy_(a_cpu)
            b.copy_(b_cpu)
            expected = a_cpu.float().add(b_cpu.float()).to(dtype)
            order = list(args.backends)
            if not args.no_reference and not args.profile:
                order.append(REFERENCE)
            rng.shuffle(order)
            for backend in order:
                row = {
                    "kernel": "vector_add",
                    "shape": [n],
                    "dtype": dtype_name,
                    "implementation": backend,
                    "scope": args.scope,
                }
                if backend == REFERENCE:
                    dst = None if args.scope == "api" else torch.empty_like(a)

                    def fn(dst=dst, src=a):
                        return src.clone() if dst is None else dst.copy_(src)

                    def sync(_):
                        torch.cuda.synchronize(args.device)

                    actual = None
                elif backend == "jax":
                    jax = importlib.import_module("jax")
                    impl = importlib.import_module("problems.vector_add.jax.implementation")
                    # DLPack preserves device identity, dtype and exact input values.
                    ja, jb = jax.dlpack.from_dlpack(a), jax.dlpack.from_dlpack(b)
                    if any(d.platform != "gpu" for d in ja.devices()):
                        raise RuntimeError("JAX did not receive GPU buffers")
                    fn = partial(impl.add, ja, jb)

                    def sync(value):
                        value.block_until_ready()

                    result = fn().block_until_ready()
                    actual = torch.from_dlpack(result).cpu()
                else:
                    out = (
                        None
                        if args.scope == "api"
                        else torch.empty(n + args.offset, dtype=dtype, device="cuda")[args.offset :]
                    )
                    fn = partial(
                        api.add,
                        a,
                        b,
                        backend=backend,
                        out=out,
                        threads=args.threads,
                        block_size=args.block_size,
                        num_warps=args.num_warps,
                    )

                    def sync(_):
                        torch.cuda.synchronize(args.device)

                    actual = fn().cpu()
                if backend == REFERENCE:
                    torch.testing.assert_close(fn(), a, rtol=0, atol=0)
                else:
                    rtol, atol = api.TOLERANCES[dtype]
                    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                    row["correctness"] = {"passed": True, "rtol": rtol, "atol": atol}
                if args.profile:
                    for _ in range(args.warmup):
                        result = fn()
                    sync(result)
                    torch.cuda.cudart().cudaProfilerStart()
                    try:
                        for _ in range(args.samples):
                            result = fn()
                        sync(result)
                    finally:
                        torch.cuda.cudart().cudaProfilerStop()
                    print("Profile capture complete; no benchmark timings produced.")
                    return
                values = measure(fn, sync, args.scope, args.warmup, args.samples)
                row.update(status="ok", raw_latency_us=values, latency_us=summarize(values))
                latency = row["latency_us"]["median"]
                streams = 2 if backend == REFERENCE else 3
                row["logical_bytes"] = n * a.element_size() * streams
                row["effective_gb_s"] = row["logical_bytes"] / latency / 1000
                row["elements_s"] = n / latency * 1e6
                if args.peak_gb_s:
                    row["fraction_of_peak"] = row["effective_gb_s"] / args.peak_gb_s
                if backend in ("cuda_vec", "cuda_vec_grid_stride"):
                    # Same rule as the launcher, applied to the buffers actually used.
                    used = out if out is not None else fn()
                    aligned = (a.data_ptr() | b.data_ptr() | used.data_ptr()) % 16 == 0
                    row["cuda_vector_path"] = "vector" if aligned else "scalar_fallback"
                rows.append(row)
                print(f"{backend:18} {dtype_name} n={n:<10} {latency:.3f} us ({args.scope})")
    for row in rows:
        if row["status"] != "ok":
            continue
        reference = next(
            (
                r
                for r in rows
                if r.get("implementation") == REFERENCE
                and r["shape"] == row["shape"]
                and r["dtype"] == row["dtype"]
            ),
            None,
        )
        if reference and row is not reference:
            row["fraction_of_copy"] = row["effective_gb_s"] / reference["effective_gb_s"]
        baseline = next(
            (
                r
                for r in rows
                if r.get("implementation") == "pytorch"
                and r["shape"] == row["shape"]
                and r["dtype"] == row["dtype"]
            ),
            None,
        )
        if baseline and row["implementation"] != REFERENCE:
            row["speedup_vs_pytorch"] = (
                baseline["latency_us"]["median"] / row["latency_us"]["median"]
            )
    settings = vars(args).copy()
    settings["output"] = str(args.output)
    settings.update(
        cache_policy="reused buffers; no flushing",
        copy_reference=None if args.no_reference else f"{REFERENCE}: device-to-device Tensor.copy_",
        allocation_policy=(
            "preallocated output" if args.scope == "device" else "output allocation included"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"schema_version": 1, "environment": metadata, "settings": settings, "results": rows},
            indent=2,
        )
        + "\n"
    )


def main():
    p = parser()
    args = p.parse_args()
    if min(args.sizes) < 1 or args.offset < 0 or args.warmup < 1 or args.samples < 1:
        p.error("sizes/warmup/samples must be positive; offset must be nonnegative")
    if "jax" in args.backends and (args.scope != "api" or args.offset != 0 or args.profile):
        p.error("JAX requires --scope api, --offset 0, and no --profile")
    if args.profile and (len(args.backends) != 1 or len(args.sizes) != 1 or len(args.dtypes) != 1):
        p.error("profiling requires exactly one backend, size and dtype")
    run(args)


if __name__ == "__main__":
    main()
