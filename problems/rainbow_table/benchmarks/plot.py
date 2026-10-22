"""Plot median and range of three run medians; never call integer work FLOPs."""

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/rainbow_table")
    groups = {}
    for path in sorted(root.glob("device_*.json")):
        data = json.loads(path.read_text())
        for row in data["results"]:
            c = row["case"]
            if row["status"] != "ok" or c["n"] != 10_000_000:
                continue
            b = row["implementation"]
            if b in ("copy", "copy_reference"):
                continue
            groups.setdefault(b, {}).setdefault(c["rounds"], []).append(
                c["n"] * c["rounds"] / row["latency_us"]["median"] / 1e6
            )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})
    for index, (b, points) in enumerate(groups.items()):
        if b == "cuda":  # documented alias of cuda_u32
            continue
        xs = sorted(points)
        ys = [statistics.median(points[x]) for x in xs]
        color = (
            axes[0]
            .plot(xs, ys, "o-", label=b, markersize=3, color=plt.get_cmap("tab20")(index))[0]
            .get_color()
        )
        axes[0].fill_between(
            xs, [min(points[x]) for x in xs], [max(points[x]) for x in xs], color=color, alpha=0.12
        )
        if b not in ("pytorch", "pytorch_int32"):
            axes[1].plot([x / 8 for x in xs], ys, "o-", color=color, markersize=3)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Rounds R",
        ylabel="Trillion hash rounds / s",
        title="R sweep · N = 10 million · RTX 5080",
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Logical hash rounds / byte (R / 8)",
        ylabel="Trillion hash rounds / s",
        title="Logical integer roofline view",
    )
    xs = [1 / 8, 100 / 8]
    axes[1].plot(xs, [960e9 * x / 1e12 for x in xs], "k--", label="960 GB/s DRAM bound")
    ceiling = max(
        statistics.median(p[100])
        for b, p in groups.items()
        if b not in ("pytorch", "pytorch_int32") and 100 in p
    )
    axes[1].axhline(
        ceiling, color="gray", linestyle=":", label=f"Best measured R=100: {ceiling:.2f} T rounds/s"
    )
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Three runs: median of medians; shading = min–max · reused input, allocation included",
        fontsize=10,
    )
    fig.tight_layout()
    target = Path("problems/rainbow_table/figures/r_sweep.svg")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    fig.savefig(root / "r_sweep.png", dpi=140)


if __name__ == "__main__":
    main()
