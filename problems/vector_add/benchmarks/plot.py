"""Plot effective bandwidth against working-set size from size-sweep JSON files.

python -m problems.vector_add.benchmarks.plot sizes_fp32.json sizes_fp16.json \
    --output-dir problems/vector_add/figures

Writes light and dark SVG variants for a README <picture> element.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SERIES = ("pytorch", "cuda_scalar", "cuda_vec", "triton")
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500"),
    },
}


def load(path):
    data = json.loads(Path(path).read_text())
    env, settings = data["environment"], data["settings"]
    curves = {}
    for row in data["results"]:
        if row["status"] == "ok" and row["implementation"] in SERIES:
            curves.setdefault(row["implementation"], []).append(
                (row["logical_bytes"], row["effective_gb_s"])
            )
    dtype = data["results"][0]["dtype"]
    return dtype, {k: sorted(v) for k, v in curves.items()}, env, settings


def size_label(value, _):
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            return f"{value / scale:g} {unit}"
    return f"{value:g} B"


def plot(panels, theme, output):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2), sharey=True)
    fig.patch.set_facecolor(t["surface"])
    axes = [axes] if len(panels) == 1 else axes
    _, _, env, settings = panels[0]
    l2, peak = env.get("l2_bytes"), settings.get("peak_gb_s")
    for ax, (dtype, curves, _, _) in zip(axes, panels, strict=True):
        ax.set_facecolor(t["surface"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.grid(True, which="major", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=t["muted"], length=0, labelsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(size_label))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        if peak:
            ax.axhline(peak, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)))
            ax.text(1.3e4, peak * 1.12, f"Spec peak {peak:g} GB/s", color=t["muted"], fontsize=8.5)
        if l2:
            ax.axvline(l2, color=t["muted"], linewidth=1, linestyle=(0, (1, 2.5)))
            ax.text(l2 * 1.12, 1.6, f"L2 {l2 / 2**20:g} MiB", color=t["muted"], fontsize=8.5)
            for x, label in ((3e4, "launch-bound"), (4e6, "L2-resident"), (l2 * 1.6, "DRAM")):
                ax.text(x, 0.45, label, color=t["muted"], fontsize=8.5, style="italic")
        for name, color in zip(SERIES, t["series"], strict=True):
            if name in curves:
                x, y = zip(*curves[name], strict=True)
                ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4.5, label=name)
        ax.set_title(dtype.upper(), loc="left", color=t["text"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Working set (A + B + C)", color=t["muted"], fontsize=9)
    axes[0].set_ylabel("Effective GB/s (log)", color=t["muted"], fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.935),
        ncol=len(labels),
        frameon=False,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(t["text"])
    fig.suptitle(
        f"Vector addition bandwidth vs size · {env['gpu']}",
        x=0.012,
        ha="left",
        color=t["text"],
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(output, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("problems/vector_add/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = [load(p) for p in args.inputs]
    for theme in THEMES:
        plot(panels, theme, args.output_dir / f"bandwidth_vs_size_{theme}.svg")
