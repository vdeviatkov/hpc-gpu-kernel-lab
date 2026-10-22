"""Keep compact, reviewable evidence alongside the docs; raw captures stay in artifacts."""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "launch__registers_per_thread",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "sm__inst_executed_pipe_alu.sum",
    "sm__inst_executed_pipe_aluheavy.sum",
    "sm__inst_executed_pipe_fma.sum",
    "sm__inst_executed_pipe_fmaheavy.sum",
    "sm__inst_executed_pipe_fmalite.sum",
)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/rainbow_table")
    groups = defaultdict(list)
    environments = []
    for path in sorted(root.glob("device_*.json")):
        data = json.loads(path.read_text())
        environments.append(
            {
                k: data["environment"].get(k)
                for k in (
                    "timestamp_utc",
                    "gpu",
                    "sm_count",
                    "cuda_runtime",
                    "cuda_toolkit",
                    "nvidia_smi",
                )
            }
        )
        for r in data["results"]:
            if r["status"] != "ok":
                raise ValueError(f"Failed benchmark: {r}")
            groups[(r["implementation"], r["case"]["n"], r["case"]["rounds"])].append(
                r["latency_us"]["median"]
            )
    bench = [
        dict(
            backend=b,
            n=n,
            rounds=r,
            run_medians_us=vals,
            median_us=statistics.median(vals),
            trillion_rounds_s=n * r / statistics.median(vals) / 1e6,
        )
        for (b, n, r), vals in sorted(groups.items())
    ]
    profiles = {}
    for path in sorted((root / "ncu").glob("*.csv")):
        rows = list(csv.reader(path.open()))
        if len(rows) < 3 or rows[0][0] != "ID":
            raise ValueError(f"Invalid profiler export: {path}")
        headers, units = rows[:2]
        selected = []
        for row in rows[2:]:
            values = dict(zip(headers, row, strict=True))
            selected.append(
                dict(
                    kernel=values["Kernel Name"],
                    metrics={
                        m: dict(value=values[m], unit=units[headers.index(m)])
                        for m in METRICS
                        if m in values
                    },
                )
            )
        profiles[path.stem] = selected
    target = Path("problems/rainbow_table/benchmarks/measured_summary.json")
    target.write_text(
        json.dumps(dict(environment=environments, benchmarks=bench, profiles=profiles), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
