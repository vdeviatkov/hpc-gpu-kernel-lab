"""Convert one real run's JSON to Markdown on stdout."""

import argparse
import json
from pathlib import Path


def markdown(data):
    lines = [
        "| Backend | N | Dtype | Scope | p20 µs | p50 µs | p80 µs | Effective GB/s "
        "| % of copy | Speedup |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in data["results"]:
        if r["status"] != "ok":
            continue
        t = r["latency_us"]
        speedup = f"{r['speedup_vs_pytorch']:.2f}×" if "speedup_vs_pytorch" in r else "—"
        copy = r.get("fraction_of_copy")
        copy = f"{copy:.1%}" if copy is not None else "—"
        lines.append(
            f"| {r['implementation']} | {r['shape'][0]} | {r['dtype']} | {r['scope']} "
            f"| {t['p20']:.3f} | {t['median']:.3f} | {t['p80']:.3f} "
            f"| {r['effective_gb_s']:.2f} | {copy} | {speedup} |"
        )
    skipped = [r for r in data["results"] if r["status"] != "ok"]
    for row in skipped:
        lines.append(f"\nSkipped {row['dtype']} {row['shape']}: {row['reason']}")
    if len(lines) == 2:
        lines.append("\nResults pending hardware benchmark")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(markdown(json.loads(args.input.read_text())))
