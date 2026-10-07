"""Plot each backend's speed as a fraction of a device copy, N = 25M, mixed signs.

python -m problems.relu.benchmarks.plot artifacts/relu/device_*.json \
    --output problems/relu/figures/relu_vs_copy.svg

A dot plot rather than bars: the differences are a few percent, and bars would need
either a zero baseline that hides them or a cut axis that exaggerates them.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

BACKENDS = ("pytorch", "cuda_select", "cuda_branch", "cuda_fmax", "cuda_vec", "triton")
SERIES = (("fp32", "FP32", "#2a78d6", "o"), ("fp16", "FP16", "#eb6834", "D"))
STYLE = {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}
CASE = {"n": 25_000_000, "dist": "mixed"}


def fractions(paths):
    """Median over runs of each backend's fraction_of_copy, per dtype."""
    values = {}
    for path in paths:
        for row in json.loads(Path(path).read_text())["results"]:
            if row.get("status") == "ok" and row["case"] == CASE and "fraction_of_copy" in row:
                key = (row["implementation"], row["dtype"])
                values.setdefault(key, []).append(row["fraction_of_copy"])
    return {key: statistics.median(v) for key, v in values.items()}, json.loads(
        Path(paths[0]).read_text()
    )["environment"]["gpu"]


def plot(data, gpu, output):
    t = STYLE
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    rows = list(reversed(BACKENDS))
    for offset, (dtype, label, color, marker) in zip((0.13, -0.13), SERIES, strict=True):
        points = [(data[(b, dtype)], i + offset) for i, b in enumerate(rows) if (b, dtype) in data]
        x, y = zip(*points, strict=True)
        ax.scatter(
            x,
            y,
            s=46,
            color=color,
            marker=marker,
            label=label,
            zorder=3,
            edgecolors=t["surface"],
            linewidths=1.5,
        )
    # Exact values in a right-hand column, so labels never sit on the reference line.
    columns = ((1.07, "FP32", "fp32"), (1.20, "FP16", "fp16"))
    for x, header, dtype in columns:
        ax.text(
            x,
            len(rows) - 0.4,
            header,
            transform=ax.get_yaxis_transform(),
            ha="right",
            fontsize=8.5,
            color=t["text"],
            fontweight="bold",
        )
        for i, b in enumerate(rows):
            if (b, dtype) in data:
                ax.text(
                    x,
                    i,
                    f"{data[(b, dtype)]:.1%}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=8.5,
                    color=t["muted"],
                )
    ax.axvline(1.0, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.997, len(rows) - 0.4, "copy speed", ha="right", fontsize=8, color=t["muted"])
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlim(0.8, 1.02)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(True, axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    for tick in ax.get_yticklabels():
        tick.set_color(t["text"])
    ax.set_xlabel(
        "Speed relative to a device copy of the same tensor (higher is better)",
        color=t["muted"],
        fontsize=9,
    )
    legend = ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=9,
        ncol=2,
        bbox_to_anchor=(0, 1.0),
        handletextpad=0.3,
    )
    for text in legend.get_texts():
        text.set_color(t["text"])
    fig.suptitle(
        f"ReLU, N = 25M · {gpu}",
        x=0.012,
        ha="left",
        color=t["text"],
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.84, 0.95))
    fig.savefig(output, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("problems/relu/figures/relu_vs_copy.svg")
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(*fractions(args.inputs), args.output)
