"""Plot each backend's speed as a fraction of a device copy, N = 25M (the source's size).

python -m problems.interleave_arrays.benchmarks.plot artifacts/interleave_arrays/device_*.json \
    --output problems/interleave_arrays/figures/interleave_vs_copy.svg

A dot plot with an x-axis from 0: the PyTorch baselines sit near 35% and cuda_output FP16
near 60%, so the full range is needed. Exact values are in a column on the right.
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
    "pytorch_stack",
    "cuda_input",
    "cuda_output",
    "cuda_pair_store",
    "cuda_vec",
)
SERIES = (("fp32", "FP32", "#2a78d6", "o"), ("fp16", "FP16", "#eb6834", "D"))
STYLE = {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}
N = 25_000_000


def fractions(paths):
    """Median over runs of each backend's fraction_of_copy at N, per dtype."""
    values = {}
    for path in paths:
        for row in json.loads(Path(path).read_text())["results"]:
            if row.get("status") == "ok" and row["case"]["n"] == N and "fraction_of_copy" in row:
                values.setdefault((row["implementation"], row["dtype"]), []).append(
                    row["fraction_of_copy"]
                )
    gpu = json.loads(Path(paths[0]).read_text())["environment"]["gpu"]
    return {key: statistics.median(v) for key, v in values.items()}, gpu


def plot(data, gpu, output):
    t = STYLE
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
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
    for x, header, dtype in ((1.10, "FP32", "fp32"), (1.24, "FP16", "fp16")):
        transform = ax.get_yaxis_transform()
        ax.text(
            x,
            len(rows) - 0.4,
            header,
            transform=transform,
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
                    transform=transform,
                    ha="right",
                    va="center",
                    fontsize=8.5,
                    color=t["muted"],
                )
    ax.axvline(1.0, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.99, len(rows) - 0.4, "copy speed", ha="right", fontsize=8, color=t["muted"])
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlim(0, 1.04)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.grid(True, axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    for tick in ax.get_yticklabels():
        tick.set_color(t["text"])
    ax.set_xlabel(
        "Speed relative to a device copy of both inputs (higher is better)",
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
        f"Interleave Arrays, N = 25M · {gpu}",
        x=0.012,
        ha="left",
        color=t["text"],
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.83, 0.95))
    fig.savefig(output, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("problems/interleave_arrays/figures/interleave_vs_copy.svg"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(*fractions(args.inputs), args.output)
