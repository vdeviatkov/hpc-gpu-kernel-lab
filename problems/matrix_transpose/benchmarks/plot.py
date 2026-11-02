"""Plot effective bandwidth against working-set size, with the L2 capacity marked.

python -m problems.matrix_transpose.benchmarks.plot artifacts/matrix_transpose

One panel per element width. The x-axis is the logical traffic of one call, 2·M·N·e bytes,
as in 07 · Matrix Copy's figure. memcpy and copy `cuda_vec` on the same shapes are plain
copies, not transposes: black reference lines marking the bandwidth ceiling, labelled as
such. Each point is the median of three run medians.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# A subset that answers the study's questions; every backend is tabulated in results.md.
# Categorical slots 1–7 of the lab palette, in their validated order.
SERIES = (
    ("cuda_naive_row", "#2a78d6", "o"),
    ("cuda_smem", "#eb6834", "D"),
    ("cuda_smem_pad", "#1baf7a", "s"),
    ("cuda_smem_swizzle", "#eda100", "^"),
    ("cuda_vec_ilp", "#e87ba4", "v"),
    ("pytorch", "#008300", "X"),
    ("triton", "#6250d6", "P"),
)
# Plain copies of the input, not transposes: the ceiling a transpose can at best match.
REFERENCES = (
    ("memcpy", "ceiling: memcpy (plain copy)", (0, (1, 1.5))),
    ("copy_cuda_vec", "ceiling: copy cuda_vec (07, plain copy)", (0, (5, 2))),
)
PANELS = (("fp32", "FP32"), ("fp16", "FP16"))
STYLE = {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}
PEAK_GB_S = 960
L2_BYTES = 64 * 2**20


def series(rows, backend, dtype):
    points = sorted(
        (r["logical_bytes"], r["effective_gb_s"])
        for r in rows
        if r["backend"] == backend and r["dtype"] == dtype
    )
    return [b / 1e6 for b, _ in points], [gb for _, gb in points]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/matrix_transpose")
    summary = json.loads((Path(__file__).parent / "measured_summary.json").read_text())
    rows = summary["benchmarks"]
    gpu = summary["environment"][0]["gpu"]
    top = max(r["effective_gb_s"] for r in rows) * 1.08
    t = STYLE
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.patch.set_facecolor(t["surface"])
    for ax, (dtype, title) in zip(axes, PANELS, strict=True):
        ax.set_facecolor(t["surface"])
        ax.axvspan(L2_BYTES / 1.5 / 1e6, L2_BYTES / 1e6, color=t["grid"], alpha=0.6, lw=0)
        ax.text(
            L2_BYTES / 1.22 / 1e6, top * 0.9, "64 MiB L2\nfills", ha="center", va="bottom",
            fontsize=8, color=t["muted"],
        )  # fmt: skip
        ax.axhline(PEAK_GB_S, color=t["muted"], lw=1, ls=(0, (4, 3)))
        ax.text(
            0.01, PEAK_GB_S + 40, "960 GB/s DRAM peak", fontsize=8, color=t["muted"],
            transform=ax.get_yaxis_transform(),
        )  # fmt: skip
        for backend, label, dash in REFERENCES:
            x, y = series(rows, backend, dtype)
            ax.plot(x, y, color=t["text"], lw=2 if dash[1][0] < 2 else 1.4, ls=dash, label=label)
        for backend, color, marker in SERIES:
            x, y = series(rows, backend, dtype)
            ax.plot(
                x, y, color=color, lw=2, marker=marker, ms=6, mec=t["surface"], mew=1.2,
                label=backend,
            )  # fmt: skip
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontsize=10, color=t["text"])
        ax.set_xlabel("Logical traffic per call, 2·M·N·e (MB, log scale)", color=t["muted"])
        ax.grid(True, which="major", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    axes[0].set_ylabel("Effective bandwidth, GB/s (higher is better)", color=t["muted"])
    axes[0].set_ylim(0, top)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, 0), handlelength=2.4,
    )  # fmt: skip
    for text in legend.get_texts():
        text.set_color(t["text"])
    fig.suptitle(
        f"Matrix transpose on {gpu}: N² = 512² to 8192² plus 7000×6000, median of three runs, "
        "inputs reused, allocation included",
        fontsize=10,
        color=t["text"],
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    target = Path(__file__).parents[1] / "figures" / "bandwidth_vs_size.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, facecolor=t["surface"])
    fig.savefig(root / "bandwidth_vs_size.png", dpi=140, facecolor=t["surface"])


if __name__ == "__main__":
    main()
