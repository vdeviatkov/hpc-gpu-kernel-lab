"""Convert one benchmark run's JSON to a Markdown table on stdout.

python -m lab.report artifacts/relu.json
"""

import argparse
import json
from pathlib import Path


def _case(row):
    if "case" in row:
        return ", ".join(f"{k}={v}" for k, v in row["case"].items())
    return f"N={row['shape'][0]}"


def _percent(row, key):
    return f"{row[key]:.1%}" if key in row else "—"


def markdown(data):
    ok = [r for r in data["results"] if r["status"] == "ok"]
    copy = any("fraction_of_copy" in r for r in ok)
    flops = any("tflop_s" in r for r in ok)
    header = ["Backend", "Case", "Dtype", "Scope", "p20 µs", "p50 µs", "p80 µs", "GB/s"]
    header += ["TFLOP/s"] * flops + ["% of copy"] * copy + ["Speedup"]
    lines = ["| " + " | ".join(header) + " |", "|---|---|---|---|" + "---:|" * (len(header) - 4)]
    for r in ok:
        t = r["latency_us"]
        cells = [
            r["implementation"],
            _case(r),
            r["dtype"],
            r["scope"],
            f"{t['p20']:.3f}",
            f"{t['median']:.3f}",
            f"{t['p80']:.3f}",
            f"{r['effective_gb_s']:.2f}",
        ]
        if flops:
            cells.append(f"{r['tflop_s']:.2f}" if "tflop_s" in r else "—")
        if copy:
            cells.append(_percent(r, "fraction_of_copy"))
        speedup = r.get("speedup_vs_pytorch")
        cells.append(f"{speedup:.2f}×" if speedup else "—")
        lines.append("| " + " | ".join(cells) + " |")
    for row in data["results"]:
        if row["status"] != "ok":
            lines.append(
                f"\n{row['status']}: {row.get('implementation', '')} {row['dtype']} "
                f"{_case(row)} — {row.get('reason', '')}"
            )
    if not ok:
        lines.append("\nResults pending hardware benchmark")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(markdown(json.loads(args.input.read_text())))
