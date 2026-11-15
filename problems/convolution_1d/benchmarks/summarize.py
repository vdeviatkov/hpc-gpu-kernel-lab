"""Keep compact, reviewable evidence alongside the docs; raw captures stay in artifacts.

Usage: python -m problems.convolution_1d.benchmarks.summarize [artifact_dir]

Reads the three runs of each sweep (ksweep_*, nsweep_*, rsweep_*, copy_* and fma_peak_*
JSON from benchmarks/run.py and fma_peak.py) and the Nsight Compute raw-page exports
(ncu/*.csv, halo/*.csv); writes benchmarks/measured_summary.json and prints the Markdown
tables used in docs/results.md. Every value is the median of the three run medians.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ORDER = (
    "cuda_naive",
    "cuda_const",
    "cuda_smem",
    "cuda_smem_reg",
    "cuda_large_k",
    "cuda_unrolled",
    "triton",
    "pytorch",
)
REG_OUTPUTS = (2, 4, 8, 16)
# The metrics sweep.sh collects (main set and halo set); other ncu columns are dropped.
KEEP = {
    "gpu__time_duration.sum", "sm__cycles_elapsed.avg.per_second", "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active", "l1tex__lsuin_requests.sum",
    "l1tex__lsuin_requests.avg.pct_of_peak_sustained_active", "l1tex__data_pipe_lsu_wavefronts.sum",
    "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_active",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "smsp__inst_executed_pipe_fma.sum",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", "smsp__inst_executed.sum",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "launch__registers_per_thread", "launch__occupancy_limit_registers",
    "sm__maximum_warps_per_active_cycle_pct", "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__grid_size", "launch__block_size", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "lts__t_sectors_srcunit_tex_op_read.sum",
}  # fmt: skip
THREADS = 256  # kThreads in cuda/kernels.cu
CHUNK = 1024  # kChunk
SCALE = {
    "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "ns": 1e-3, "us": 1, "usecond": 1,
    "ms": 1e3, "msecond": 1e3, "Ghz": 1, "Mhz": 1e-3,
}  # fmt: skip


def number(text):
    return float(text.replace(",", ""))


def element_size(dtype):
    return 4 if dtype == "fp32" else 2


def load(root, pattern):
    """{(backend, dtype, n, k): [run medians]} and {key: logical_bytes} for one sweep."""
    medians, nbytes, environments = defaultdict(list), {}, []
    for path in sorted(root.glob(pattern)):
        data = json.loads(path.read_text())
        env = data["environment"]
        environments.append(
            {k: env.get(k) for k in ("timestamp_utc", "gpu", "sm_count", "l2_bytes", "commit")}
            | {"file": path.name, "torch": env["packages"].get("torch")}
            | {"triton": env["packages"].get("triton"), "cuda_runtime": env.get("cuda_runtime")}
        )
        for r in data["results"]:
            if r["status"] == "not_implemented":
                continue
            if r["status"] != "ok":
                raise ValueError(f"Failed benchmark in {path.name}: {r}")
            case = r["case"]
            backend = "memcpy" if r["implementation"] == "copy_reference" else r["implementation"]
            if "m" in case:  # 07 · Matrix Copy on an (N / 1024) x 1024 matrix
                key = (f"copy_{backend}", r["dtype"], case["m"] * case["n"], None)
            else:
                key = (backend, r["dtype"], case["n"], case["k"])
            medians[key].append(r["latency_us"]["median"])
            nbytes[key] = r["logical_bytes"]
    return medians, nbytes, environments


def rows_for(sweep, medians, nbytes):
    rows = []
    for (backend, dtype, n, k), values in sorted(medians.items(), key=str):
        median = statistics.median(values)
        row = dict(
            sweep=sweep,
            backend=backend,
            dtype=dtype,
            n=n,
            k=k,
            run_medians_us=values,
            median_us=median,
            spread=(max(values) - min(values)) / median,
            logical_bytes=nbytes[(backend, dtype, n, k)],
        )
        row["effective_gb_s"] = row["logical_bytes"] / median / 1e3
        if k is not None and not backend.startswith(("memcpy", "copy_")):
            row["flops"] = 2 * k * (n - k + 1)
            row["tflop_s"] = row["flops"] / median / 1e6
            row["intensity_flop_per_byte"] = row["flops"] / row["logical_bytes"]
        rows.append(row)
    return rows


def peaks(root):
    runs = [json.loads(p.read_text()) for p in sorted(root.glob("fma_peak_*.json"))]
    if not runs:
        return None
    return dict(
        runs=runs,
        achieved_tflop_s=statistics.median(r["achieved_tflop_s"] for r in runs),
        sm_clock_mhz=statistics.median(r["sm_clock_mhz_median"] for r in runs),
        theoretical_tflop_s=statistics.median(r["theoretical_tflop_s_at_clock"] for r in runs),
    )


def profiles(directory):
    result = {}
    for path in sorted(directory.glob("*.csv")):
        rows = list(csv.reader(path.open()))
        if len(rows) < 3 or rows[0][0] != "ID":
            raise ValueError(f"Invalid profiler export: {path}")
        headers, units = rows[:2]
        # cuda_const converts w to FP32 with a small kernel before the convolution (no-op in
        # FP32): keep the longest kernel in the capture, the convolution itself.
        time = headers.index("gpu__time_duration.sum")
        main_row = max(rows[2:], key=lambda row: number(row[time]) * SCALE.get(units[time], 1))
        values = dict(zip(headers, main_row, strict=True))
        metrics = {}
        for name, unit in zip(headers, units, strict=True):
            if name not in KEEP or values[name] in ("", "n/a"):
                continue
            try:
                metrics[name] = number(values[name]) * SCALE.get(unit, 1)
            except ValueError:
                continue
        result[path.stem] = dict(
            kernel=values["Kernel Name"], kernels=len(rows) - 2, metrics=metrics
        )
    return result


def table(header, align, lines):
    return "\n".join([header, align, *lines])


def ksweep_table(rows, dtype, ks):
    lookup = {(r["backend"], r["k"]): r for r in rows if r["dtype"] == dtype}
    out = []
    for backend in ORDER:
        if not any((backend, k) in lookup for k in ks):
            continue
        cells = []
        for k in ks:
            r = lookup.get((backend, k))
            cells.append(f"{r['median_us']:,.0f} / {r['tflop_s']:.1f}" if r else "—")
        out.append(f"| `{backend}` | " + " | ".join(cells) + " |")
    header = f"| {dtype.upper()} µs / TFLOP/s | " + " | ".join(f"K={k}" for k in ks) + " |"
    return table(header, "|---|" + "---:|" * len(ks), out)


def ksweep_gb_table(rows, copy_rows, dtype, ks, n):
    lookup = {(r["backend"], r["k"]): r for r in rows if r["dtype"] == dtype}
    copy = next(r for r in copy_rows if r["dtype"] == dtype and r["n"] == n)
    out = [f"| copy `cuda_vec` (ceiling) | {copy['effective_gb_s']:.0f} |" + " |" * (len(ks) - 1)]
    for backend in ORDER:
        if not any((backend, k) in lookup for k in ks):
            continue
        cells = [
            f"{lookup[(backend, k)]['effective_gb_s']:.0f}" if (backend, k) in lookup else "—"
            for k in ks
        ]
        out.append(f"| `{backend}` | " + " | ".join(cells) + " |")
    header = f"| {dtype.upper()} GB/s | " + " | ".join(f"K={k}" for k in ks) + " |"
    return table(header, "|---|" + "---:|" * len(ks), out)


def nsweep_table(rows, copy_rows, dtype):
    ns = sorted({r["n"] for r in rows})
    lookup = {(r["backend"], r["n"]): r for r in rows if r["dtype"] == dtype}
    copies = {
        r["n"]: r for r in copy_rows if r["dtype"] == dtype and r["backend"] == "copy_cuda_vec"
    }
    out = []
    for backend in ("copy_cuda_vec", *ORDER):
        source = copies if backend == "copy_cuda_vec" else None
        cells = []
        for n in ns:
            r = source.get(n) if source is not None else lookup.get((backend, n))
            cells.append(f"{r['effective_gb_s']:.0f}" if r else "—")
        if all(c == "—" for c in cells):
            continue
        label = "copy `cuda_vec` (ceiling)" if backend == "copy_cuda_vec" else f"`{backend}`"
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    header = f"| {dtype.upper()} GB/s, K=31 | " + " | ".join(f"{n / 2**20:g} M" for n in ns) + " |"
    return table(header, "|---|" + "---:|" * len(ns), out)


def rsweep_table(rows, profiles_):
    out = []
    for dtype in ("fp32", "fp16"):
        for r in REG_OUTPUTS:
            cells = []
            for k in (31, 255, 2047):
                row = next(
                    (
                        x
                        for x in rows
                        if (x["backend"], x["dtype"], x["k"]) == (f"cuda_smem_reg_r{r}", dtype, k)
                    ),
                    None,
                )
                cells.append(f"{row['median_us']:,.0f} / {row['tflop_s']:.1f}" if row else "—")
            p = profiles_.get(f"cuda_smem_reg_r{r}_{dtype}_k255", {}).get("metrics", {})
            names = (
                "launch__registers_per_thread",
                "sm__maximum_warps_per_active_cycle_pct",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
            )
            cells += [f"{p.get(name, float('nan')):.0f}" for name in names]
            out.append(f"| {dtype.upper()} | {r} | " + " | ".join(cells) + " |")
    header = (
        "| dtype | R | K=31 µs / TFLOP/s | K=255 | K=2047 | Regs | Theoretical occ. % "
        "| Achieved occ. % | FMA pipe % (K=255) |"
    )
    return table(header, "|---|---:|" + "---:|" * 7, out)


def ncu_table(profiles_, dtype, k, n):
    outputs = n - k + 1
    fmas = outputs * k
    out = []
    for backend in ORDER:
        p = profiles_.get(f"{backend}_{dtype}_k{k}")
        if not p:
            continue
        m = p["metrics"]

        def get(name, m=m):
            return m.get(name, float("nan"))

        shared_wf = get("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum")
        out.append(
            f"| `{backend}` | {get('gpu__time_duration.sum'):,.0f} "
            f"| {(get('dram__bytes_op_read.sum') + get('dram__bytes_op_write.sum')) / outputs:.2f} "
            f"| {get('l1tex__t_sector_hit_rate.pct'):.1f} "
            f"| {get('l1tex__throughput.avg.pct_of_peak_sustained_active'):.0f} "
            f"| {get('l1tex__lsuin_requests.sum') / (fmas / 32):.3f} "
            f"| {shared_wf / (fmas / 32):.3f} "
            f"| {get('sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active'):.0f} "
            f"| {get('smsp__inst_executed.sum') / get('smsp__inst_executed_pipe_fma.sum'):.2f} "
            f"| {get('smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio'):.2f} "
            f"| {get('smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio'):.2f} "
            f"| {get('smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio'):.2f} "
            f"| {get('launch__registers_per_thread'):.0f} "
            f"| {get('sm__warps_active.avg.pct_of_peak_sustained_active'):.0f} |"
        )
    header = (
        f"| {dtype.upper()} K={k} | µs | DRAM B/output | L1 hit % | L1 % | LSU req / warp-FMA "
        "| shared ld wavefr / warp-FMA | FMA pipe % | inst / FMA inst | LG thr | MIO thr "
        "| barrier | Regs | Warps active % |"
    )
    return table(header, "|---|" + "---:|" * 13, out)


def halo_table(halo):
    """cuda_large_k global x+w elements loaded per output against the model.

    A block of span = 256·R outputs loads span + taps x elements per 1024-tap chunk and its
    K filter taps once, so per output: chunks + (K + K)/span, about 1 + (K - 1)/span (halo)
    + K/span (filter) for K <= 1024. Measured: global load sectors · 32 B / 4 B / outputs.
    """
    out = []
    for name, p in sorted(halo.items()):
        _, r, k = name.split("_")  # halo_r{R}_k{K}
        r, k = int(r[1:]), int(k[1:])
        span = THREADS * r
        chunks = -(-k // CHUNK)
        outputs = (1 << 24) - k + 1
        model = chunks + 2 * k / span
        sectors = p["metrics"].get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", float("nan"))
        measured = sectors * 32 / 4 / outputs
        out.append(
            f"| {r} | {span} | {k} | {(k - 1) / span:.3f} | {model:.3f} | {measured:.3f} "
            f"| {measured / model:.3f} |"
        )
    header = (
        "| R | span (outputs/block) | K | halo (K−1)/span | model elements/output "
        "| measured | measured / model |"
    )
    return table(header, "|---:|---:|---:|---:|---:|---:|---:|", out)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/convolution_1d")
    environments, rows = [], []
    sweeps = {}
    for sweep in ("ksweep", "nsweep", "rsweep", "copy"):
        medians, nbytes, env = load(root, f"{sweep}_*.json")
        environments += env
        sweeps[sweep] = rows_for(sweep, medians, nbytes)
        rows += sweeps[sweep]
    peak = peaks(root)
    profiles_ = profiles(root / "ncu")
    halo = profiles(root / "halo") if (root / "halo").exists() else {}
    target = Path(__file__).parent / "measured_summary.json"
    target.write_text(
        json.dumps(
            dict(
                environment=environments,
                fma_peak=peak,
                benchmarks=rows,
                profiles=profiles_,
                halo=halo,
            ),
            indent=2,
        )
        + "\n"
    )
    ks = sorted({r["k"] for r in sweeps["ksweep"]})
    n_k = sweeps["ksweep"][0]["n"]
    if peak:
        print(
            f"FMA peak: achieved {peak['achieved_tflop_s']:.2f} TFLOP/s at "
            f"{peak['sm_clock_mhz']:.0f} MHz (theoretical {peak['theoretical_tflop_s']:.2f})"
        )
    for dtype in ("fp32", "fp16"):
        print(f"\n### K sweep {dtype}, N = {n_k}\n\n{ksweep_table(sweeps['ksweep'], dtype, ks)}")
        print(f"\n{ksweep_gb_table(sweeps['ksweep'], sweeps['copy'], dtype, ks, n_k)}")
    for dtype in ("fp32", "fp16"):
        print(f"\n### N sweep {dtype}\n\n{nsweep_table(sweeps['nsweep'], sweeps['copy'], dtype)}")
    print(f"\n### R sweep\n\n{rsweep_table(sweeps['rsweep'], profiles_)}")
    for dtype in ("fp32", "fp16"):
        for k in (3, 31, 255, 2047):
            print(f"\n### ncu {dtype} K={k}\n\n{ncu_table(profiles_, dtype, k, 1 << 24)}")
    if halo:
        print(f"\n### halo\n\n{halo_table(halo)}")
    spreads = sorted(r["spread"] for r in rows)
    print(
        f"\nrun-median spread: median {statistics.median(spreads):.1%}, "
        f"max {spreads[-1]:.1%}, {sum(s > 0.05 for s in spreads)} of {len(spreads)} over 5%"
    )


if __name__ == "__main__":
    main()
