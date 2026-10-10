"""Plot each backend's speed as a fraction of a device copy at N = 100M (DRAM-bound).

python -m problems.reverse_array.benchmarks.plot artifacts/reverse_array/device_*.json \
    --output problems/reverse_array/figures/reverse_vs_copy.svg

Two panels (FP32, FP16), each with aligned N (multiple of 16) and misaligned N + 1. A dot
plot rather than bars: most backends sit within a few percent of a copy.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

BACKENDS = (
    "pytorch",
    "cuda_pair",
    "cuda_vec",
    "cuda_tile",
    "cuda_tile_async",
    "triton_pair",
    "triton_flip",
)
SERIES = (
    (100_000_000, "N = 100,000,000 (aligned)", "#2a78d6", "o"),
    (100_000_001, "N = 100,000,001 (misaligned)", "#eb6834", "D"),
)
STYLE = {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}


def fractions(paths):
    """Median over runs of fraction_of_copy per (backend, dtype, n), for uniform inputs."""
    values = {}
    for path in paths:
        for row in json.loads(Path(path).read_text())["results"]:
            if row.get("status") == "ok" and "fraction_of_copy" in row:
                key = (row["implementation"], row["dtype"], row["case"]["n"])
                values.setdefault(key, []).append(row["fraction_of_copy"])
    gpu = json.loads(Path(paths[0]).read_text())["environment"]["gpu"]
    return {key: statistics.median(v) for key, v in values.items()}, gpu


def plot(data, gpu, output):
    t = STYLE
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fig.patch.set_facecolor(t["surface"])
    rows = list(reversed(BACKENDS))
    for ax, dtype in zip(axes, ("fp32", "fp16"), strict=True):
        ax.set_facecolor(t["surface"])
        for offset, (n, label, color, marker) in zip((0.13, -0.13), SERIES, strict=True):
            points = [
                (data[(b, dtype, n)], i + offset)
                for i, b in enumerate(rows)
                if (b, dtype, n) in data
            ]
            x, y = zip(*points, strict=True)
            ax.scatter(
                x,
                y,
                s=44,
                color=color,
                marker=marker,
                label=label,
                zorder=3,
                edgecolors=t["surface"],
                linewidths=1.5,
            )
        ax.axvline(1.0, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(0.995, len(rows) - 0.45, "copy speed", ha="right", fontsize=8, color=t["muted"])
        ax.set_xlim(0.45, 1.03)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.grid(True, axis="x", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=t["muted"], length=0, labelsize=9)
        ax.set_title(dtype.upper(), loc="left", color=t["text"], fontsize=11, fontweight="bold")
        ax.set_xlabel(
            "Speed relative to a device copy (higher is better)", color=t["muted"], fontsize=9
        )
    axes[0].set_yticks(range(len(rows)), rows)
    for tick in axes[0].get_yticklabels():
        tick.set_color(t["text"])
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=2, frameon=False
    )
    for text in legend.get_texts():
        text.set_color(t["text"])
    fig.suptitle(
        f"Reverse Array, in place · {gpu}",
        x=0.012,
        ha="left",
        color=t["text"],
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("problems/reverse_array/figures/reverse_vs_copy.svg")
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(*fractions(args.inputs), args.output)
