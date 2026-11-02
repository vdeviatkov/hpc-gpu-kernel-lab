"""Keep compact, reviewable evidence alongside the docs; raw captures stay in artifacts.

Usage: python -m problems.matrix_transpose.benchmarks.summarize [artifact_dir]

Reads device_*.json (lab.bench runs of every transpose kernel, with memcpy as
copy_reference), copy_*.json (07 · Matrix Copy's cuda_vec on the same cases), ncu/*.csv
(raw-page exports) and the Nsight Systems kernel summary; writes
benchmarks/measured_summary.json and prints the Markdown tables used in docs/results.md.
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
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__grid_size",
    "launch__block_size",
    "smsp__inst_executed.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
)
# memcpy is the transpose runs' copy_reference; copy_cuda_vec is 07's best copy kernel.
ORDER = (
    "memcpy",
    "copy_cuda_vec",
    "pytorch",
    "cuda_naive_row",
    "cuda_naive_col",
    "cuda_smem",
    "cuda_smem_pad",
    "cuda_smem_swizzle",
    "cuda_vec_ilp",
    "triton",
)
# Plain contiguous copies of the input (not transposes): bandwidth ceilings, listed first and
# set apart in every table so they are not read as competing transposes.
CEILINGS = ("memcpy", "copy_cuda_vec")
LABELS = {"memcpy": "memcpy", "copy_cuda_vec": "copy `cuda_vec`"}
PEAK_GB_S = 960

# ncu scales units per value (byte, Kbyte, Mbyte, ...; ns, us, ms); convert to bytes and µs.
SCALE = {
    "byte": 1, "byte/block": 1, "Kbyte/block": 1e3, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
    "ns": 1e-3, "us": 1, "usecond": 1, "ms": 1e3, "msecond": 1e3,
}  # fmt: skip


def number(text):
    return float(text.replace(",", ""))


def case_name(m, n):
    return f"{n}²" if m == n else f"{m}×{n}"


def load_runs(root, pattern, rename):
    groups = defaultdict(list)
    environments = []
    for path in sorted(root.glob(pattern)):
        data = json.loads(path.read_text())
        env = data["environment"]
        environments.append(
            {
                k: env.get(k)
                for k in ("timestamp_utc", "gpu", "sm_count", "l2_bytes", "cuda_runtime")
                + ("cuda_toolkit", "nvidia_smi", "commit")
            }
            | {"file": path.name, "torch": env["packages"].get("torch")}
        )
        for r in data["results"]:
            if r["status"] == "not_implemented":
                continue
            if r["status"] != "ok":
                raise ValueError(f"Failed benchmark: {r}")
            name = rename.get(r["implementation"])
            if name is None:
                continue
            key = (name, r["dtype"], r["case"]["m"], r["case"]["n"])
            groups[key].append((r["latency_us"]["median"], r["logical_bytes"]))
    return environments, groups


def benchmarks(root):
    env_t, transpose = load_runs(
        root,
        "device_*.json",
        {"copy_reference": "memcpy", "pytorch": "pytorch"}
        | {b: b for b in ORDER if b.startswith("cuda_") or b == "triton"},
    )
    env_c, copy = load_runs(root, "copy_*.json", {"cuda_vec": "copy_cuda_vec"})
    rows = []
    for (backend, dtype, m, n), values in sorted({**transpose, **copy}.items()):
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
    lookup = {(r["backend"], r["dtype"], r["m"], r["n"]): r["median_us"] for r in rows}
    for r in rows:
        key = (r["dtype"], r["m"], r["n"])
        r["fraction_of_peak"] = r["effective_gb_s"] / PEAK_GB_S
        r["fraction_of_memcpy"] = lookup[("memcpy", *key)] / r["median_us"]
        r["fraction_of_copy_cuda_vec"] = lookup[("copy_cuda_vec", *key)] / r["median_us"]
    return env_t + env_c, rows


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


def timeline(root):
    path = root / "nsys" / "timeline_cuda_gpu_kern_sum.csv"
    if not path.exists():
        return []
    keep = ("Time (%)", "Instances", "Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)", "Name")
    return [{k: row[k] for k in keep} for row in csv.DictReader(path.open())]


def label(backend):
    return LABELS.get(backend, f"`{backend}`")


def grouped(lines, columns):
    """Return an add(backend, cells) that prefixes each group with a bold header row."""
    previous = None

    def add(backend, cells):
        nonlocal previous
        group = (
            "Copy ceilings (plain copies, not transposes)" if backend in CEILINGS else "Transposes"
        )
        if group != previous:
            lines.append(f"| ***{group}*** |" + " |" * columns)
            previous = group
        lines.append(f"| {label(backend)} | " + cells + " |")

    return add


def latency_table(rows, dtype):
    table = {(r["backend"], r["m"], r["n"]): r for r in rows if r["dtype"] == dtype}
    shapes = sorted({(r["m"], r["n"]) for r in rows}, key=lambda s: s[0] * s[1])
    lines = [
        f"| {dtype.upper()} GB/s | " + " | ".join(case_name(*s) for s in shapes) + " |",
        "|---|" + "---:|" * len(shapes),
    ]
    add = grouped(lines, len(shapes))
    for backend in ORDER:
        cells = [
            f"{table[(backend, *s)]['effective_gb_s']:.0f}" if (backend, *s) in table else "—"
            for s in shapes
        ]
        add(backend, " | ".join(cells))
    return "\n".join(lines)


def source_table(rows, m, n):
    table = {(r["backend"], r["dtype"]): r for r in rows if (r["m"], r["n"]) == (m, n)}
    lines = [
        "| Backend | FP32 µs | GB/s | vs memcpy | vs copy `cuda_vec` "
        "| FP16 µs | GB/s | vs memcpy | vs copy `cuda_vec` |",
        "|---|" + "---:|" * 8,
    ]
    add = grouped(lines, 8)
    for backend in ORDER:
        cells = []
        for dtype in ("fp32", "fp16"):
            r = table[(backend, dtype)]
            cells += [
                f"{r['median_us']:.1f}",
                f"{r['effective_gb_s']:.0f}",
                f"{r['fraction_of_memcpy']:.0%}",
                f"{r['fraction_of_copy_cuda_vec']:.0%}",
            ]
        add(backend, " | ".join(cells))
    return "\n".join(lines)


def profile_table(profiles_, suffix):
    def metric(p, name):
        entry = p["metrics"][name]
        return number(entry["value"]) * SCALE.get(entry["unit"], 1)

    def ratio(p, a, b):
        denominator = metric(p, b)
        return metric(p, a) / denominator if denominator else 0.0

    lines = [
        "| Backend | Time µs | DRAM rd MB | DRAM wr MB | L2 hit % | ld sect/req | st sect/req "
        "| smem ld conflicts | smem st conflicts | ld wavefr/inst | st wavefr/inst | Regs "
        "| smem B/block | Warps active % |",
        "|---|" + "---:|" * 13,
    ]
    add = grouped(lines, 13)
    for backend in ORDER:
        p = profiles_.get(f"{backend}_{suffix}")
        if not p:
            continue
        add(
            backend,
            f"{metric(p, 'gpu__time_duration.sum'):.1f} "
            f"| {metric(p, 'dram__bytes_op_read.sum') / 1e6:.1f} "
            f"| {metric(p, 'dram__bytes_op_write.sum') / 1e6:.1f} "
            f"| {metric(p, 'lts__t_sector_hit_rate.pct'):.1f} "
            f"| {ratio(p, 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', 'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'):.1f} "  # noqa: E501
            f"| {ratio(p, 'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum', 'l1tex__t_requests_pipe_lsu_mem_global_op_st.sum'):.1f} "  # noqa: E501
            f"| {metric(p, 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'):,.0f} "
            f"| {metric(p, 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'):,.0f} "
            f"| {ratio(p, 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum', 'smsp__sass_inst_executed_op_shared_ld.sum'):.2f} "  # noqa: E501
            f"| {ratio(p, 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum', 'smsp__sass_inst_executed_op_shared_st.sum'):.2f} "  # noqa: E501
            f"| {metric(p, 'launch__registers_per_thread'):.0f} "
            f"| {metric(p, 'launch__shared_mem_per_block_static'):.0f} "
            f"| {metric(p, 'sm__warps_active.avg.pct_of_peak_sustained_active'):.1f}",
        )
    return "\n".join(lines)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/matrix_transpose")
    environments, rows = benchmarks(root)
    profiles_ = profiles(root)
    kernels = timeline(root)
    target = Path(__file__).parent / "measured_summary.json"
    target.write_text(
        json.dumps(
            dict(environment=environments, benchmarks=rows, profiles=profiles_, nsys=kernels),
            indent=2,
        )
        + "\n"
    )
    print(f"\n### 7000×6000\n\n{source_table(rows, 7000, 6000)}")
    for dtype in ("fp32", "fp16"):
        print(f"\n### sweep {dtype}\n\n{latency_table(rows, dtype)}")
    for dtype in ("fp32", "fp16"):
        for regime in ("7000x6000_cold", "2048_warm"):
            print(f"\n### ncu {dtype} {regime}\n\n{profile_table(profiles_, f'{dtype}_{regime}')}")


if __name__ == "__main__":
    main()
