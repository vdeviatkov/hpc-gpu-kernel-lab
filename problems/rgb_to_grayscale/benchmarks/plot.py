"""Plot each backend's effective bandwidth at 2048 x 2048 and 4096 x 4096.

python -m problems.rgb_to_grayscale.benchmarks.plot artifacts/rgb_to_grayscale/device_*.json \
    --output problems/rgb_to_grayscale/figures/rgb_bandwidth.svg

Two panels on one x-axis: at 2048 x 2048 the FP16 image stays in L2 between calls, so a fast kernel
goes past the DRAM peak (dashed line); at 4096 x 4096 every kernel is held below it.
Effective bandwidth is logical bytes (image read once, output written once) per second.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BACKENDS = (
    "pytorch",
    "pytorch_matmul",
    "pytorch_compile",
    "cuda_pixel",
    "cuda_vec",
    "cuda_smem",
    "cuda_planar",
    "triton",
)
SIZES = (
    (2048, "2048 × 2048 (source size): FP16 fits in L2"),
    (4096, "4096 × 4096: DRAM-bound"),
)
SERIES = (("fp32", "FP32", "#2a78d6", "o"), ("fp16", "FP16", "#eb6834", "D"))
STYLE = {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}
PEAK_GB_S = 960


def bandwidths(paths):
    """Median over runs of each backend's effective GB/s, per (size, backend, dtype)."""
    values = {}
    for path in paths:
        for row in json.loads(Path(path).read_text())["results"]:
            side = row["case"]["h"]
            if row.get("status") == "ok" and side == row["case"]["w"] and side in dict(SIZES):
                key = (side, row["implementation"], row["dtype"])
                values.setdefault(key, []).append(row["effective_gb_s"])
    gpu = json.loads(Path(paths[0]).read_text())["environment"]["gpu"]
    return {key: statistics.median(v) for key, v in values.items()}, gpu


def plot(data, gpu, output, subtitle=""):
    t = STYLE
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    fig.patch.set_facecolor(t["surface"])
    rows = list(reversed(BACKENDS))
    top = max(data.values())
    for ax, (side, title) in zip(axes, SIZES, strict=True):
        ax.set_facecolor(t["surface"])
        for offset, (dtype, label, color, marker) in zip((0.13, -0.13), SERIES, strict=True):
            points = [
                (data[(side, b, dtype)], i + offset)
                for i, b in enumerate(rows)
                if (side, b, dtype) in data
            ]
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
        ax.axvline(PEAK_GB_S, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(PEAK_GB_S + 25, len(rows) - 0.45, "DRAM peak", fontsize=8, color=t["muted"])
        ax.set_title(title, loc="left", fontsize=9.5, color=t["text"])
        ax.set_xlim(0, top * 1.08)
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.grid(True, axis="x", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=t["muted"], length=0, labelsize=9)
        ax.set_xlabel("Effective bandwidth, GB/s (higher is better)", color=t["muted"], fontsize=9)
    axes[0].set_yticks(range(len(rows)), rows)
    for tick in axes[0].get_yticklabels():
        tick.set_color(t["text"])
    legend = axes[0].legend(
        loc="lower left",
        frameon=False,
        fontsize=9,
        ncol=2,
        bbox_to_anchor=(0, 1.07),
        handletextpad=0.3,
    )
    for text in legend.get_texts():
        text.set_color(t["text"])
    fig.suptitle(
        f"RGB to Grayscale · {gpu}",
        x=0.012,
        ha="left",
        color=t["text"],
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    if subtitle:
        fig.text(0.012, 0.015, subtitle, fontsize=8, color=t["muted"])
    fig.savefig(output, facecolor=t["surface"])
    if output.suffix.lower() == ".svg":
        output.write_text(
            "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
        )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("problems/rgb_to_grayscale/figures/rgb_bandwidth.svg"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dates = sorted(
        {json.loads(p.read_text())["environment"]["timestamp_utc"][:10] for p in args.inputs}
    )
    subtitle = (
        f"{', '.join(dates)} UTC · median of {len(args.inputs)} runs · all backends incl. Triton"
    )
    plot(*bandwidths(args.inputs), args.output, subtitle)
