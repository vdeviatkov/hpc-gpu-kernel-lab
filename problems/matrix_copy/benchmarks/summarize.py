"""Keep compact, reviewable evidence alongside the docs; raw captures stay in artifacts.

Usage: python -m problems.matrix_copy.benchmarks.summarize [artifact_dir]

Reads device_*.json (lab.bench runs), ncu/*.csv (raw-page exports) and sass/counts.json,
writes benchmarks/measured_summary.json, and prints the Markdown tables used in
docs/results.md.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__grid_size",
    "launch__block_size",
    "smsp__inst_executed.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
)
ORDER = (
    "copy_reference",
    "pytorch",
    "cuda_2d",
    "cuda_2d_colmajor",
    "cuda_flat",
    "cuda_vec",
    "cuda_vec_ilp2",
    "cuda_vec_ilp4",
    "cuda_vec_gridstride",
    "cuda_stream",
    "triton",
    "triton_2d",
)
PEAK_GB_S = 960


# ncu scales units per value (byte, Kbyte, Mbyte, ...; ns, us, ms); convert to bytes and µs.
SCALE = {
    "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
    "ns": 1e-3, "us": 1, "usecond": 1, "ms": 1e3, "msecond": 1e3,
}  # fmt: skip


def number(text):
    return float(text.replace(",", ""))


def benchmarks(root):
    groups = defaultdict(list)
    environments = []
    for path in sorted(root.glob("device_*.json")):
        data = json.loads(path.read_text())
        env = data["environment"]
        environments.append(
            {
                k: env.get(k)
                for k in (
                    "timestamp_utc",
                    "gpu",
                    "sm_count",
                    "l2_bytes",
                    "cuda_runtime",
                    "cuda_toolkit",
                    "nvidia_smi",
                    "commit",
                )
            }
            | {p: env["packages"].get(p) for p in ("torch", "triton")}
        )
        for r in data["results"]:
            if r["status"] != "ok":
                raise ValueError(f"Failed benchmark: {r}")
            key = (r["implementation"], r["dtype"], r["case"]["m"], r["case"]["n"])
            groups[key].append((r["latency_us"]["median"], r["logical_bytes"]))
    rows = []
    for (backend, dtype, m, n), values in sorted(groups.items()):
        medians = [v for v, _ in values]
        nbytes = values[0][1]
        median = statistics.median(medians)
        rows.append(
            dict(
                backend=backend,
                dtype=dtype,
                m=m,
                n=n,
                logical_bytes=nbytes,
                run_medians_us=medians,
                median_us=median,
                effective_gb_s=nbytes / median / 1000,
            )
        )
    copy = {(r["dtype"], r["n"]): r["median_us"] for r in rows if r["backend"] == "copy_reference"}
    for r in rows:
        r["fraction_of_peak"] = r["effective_gb_s"] / PEAK_GB_S
        r["fraction_of_copy"] = copy[(r["dtype"], r["n"])] / r["median_us"]
    return environments, rows


def profiles(root):
    result = {}
    for path in sorted((root / "ncu").glob("*.csv")):
        rows = list(csv.reader(path.open()))
        if len(rows) < 3 or rows[0][0] != "ID":
            raise ValueError(f"Invalid profiler export: {path}")
        headers, units = rows[:2]
        if len(rows) != 3:
            raise ValueError(f"Expected one profiled kernel in {path}")
        values = dict(zip(headers, rows[2], strict=True))
        result[path.stem] = dict(
            kernel=values["Kernel Name"],
            metrics={
                m: dict(value=values[m], unit=units[headers.index(m)])
                for m in METRICS
                if m in values
            },
        )
    return result


def latency_table(rows, dtype, sizes):
    table = {(r["backend"], r["n"]): r for r in rows if r["dtype"] == dtype}
    lines = [
        "| " + dtype.upper() + " | " + " | ".join(f"{n}²" for n in sizes) + " |",
        "|---|" + "---:|" * len(sizes),
    ]
    for backend in ORDER:
        cells = []
        for n in sizes:
            r = table.get((backend, n))
            cells.append(f"{r['median_us']:.1f} ({r['effective_gb_s']:.0f})" if r else "—")
        lines.append(f"| `{backend}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def profile_table(profiles_, suffix):
    def metric(p, name):
        entry = p["metrics"][name]
        return number(entry["value"]) * SCALE.get(entry["unit"], 1)

    lines = [
        "| Backend | Time µs | DRAM read MB | DRAM write MB | L2 hit % | DRAM busy % | L2 busy % "
        "| ld sectors/req | st sectors/req | Warps active % | Regs | Grid |",
        "|---|" + "---:|" * 11,
    ]
    for backend in ORDER:
        p = profiles_.get(f"{backend}_{suffix}")
        if not p:
            continue
        ld = metric(p, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") / metric(
            p, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
        )
        st = metric(p, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum") / metric(
            p, "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"
        )
        lines.append(
            f"| `{backend}` | {metric(p, 'gpu__time_duration.sum'):.1f} "
            f"| {metric(p, 'dram__bytes_op_read.sum') / 1e6:.1f} "
            f"| {metric(p, 'dram__bytes_op_write.sum') / 1e6:.1f} "
            f"| {metric(p, 'lts__t_sector_hit_rate.pct'):.1f} "
            f"| {metric(p, 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'):.1f} "
            f"| {metric(p, 'lts__throughput.avg.pct_of_peak_sustained_elapsed'):.1f} "
            f"| {ld:.1f} | {st:.1f} "
            f"| {metric(p, 'sm__warps_active.avg.pct_of_peak_sustained_active'):.1f} "
            f"| {metric(p, 'launch__registers_per_thread'):.0f} "
            f"| {metric(p, 'launch__grid_size'):.0f} |"
        )
    return "\n".join(lines)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/matrix_copy")
    environments, rows = benchmarks(root)
    profiles_ = profiles(root)
    sass_path = root / "sass" / "counts.json"
    sass = json.loads(sass_path.read_text()) if sass_path.exists() else {}
    target = Path(__file__).parent / "measured_summary.json"
    target.write_text(
        json.dumps(
            dict(environment=environments, benchmarks=rows, profiles=profiles_, sass=sass),
            indent=2,
        )
        + "\n"
    )
    sizes = sorted({r["n"] for r in rows})
    for dtype in sorted({r["dtype"] for r in rows}):
        print(f"\n### {dtype}\n\n{latency_table(rows, dtype, sizes)}")
    for dtype in ("fp32", "fp16"):
        for regime in ("4096_cold", "2048_warm"):
            print(f"\n### ncu {dtype} {regime}\n\n{profile_table(profiles_, f'{dtype}_{regime}')}")


if __name__ == "__main__":
    main()
