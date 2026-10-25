"""Plot effective bandwidth against working-set size, with the L2 capacity marked.

python -m problems.matrix_copy.benchmarks.plot artifacts/matrix_copy

One panel per element width (BF16 tracks FP16). The x-axis is the logical traffic of one
call, 2·N²·e bytes, so the L2 boundary sits at the same place in both panels. Because two
output blocks alternate between calls (see cases.py), the resident footprint is 2–3 times
N²·e, and the L2 boundary is a band: from 64 MiB / 1.5 of traffic (3·N²·e fills L2) to
64 MiB (2·N²·e fills L2). Each point is the median of three run medians.
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# A subset that answers the study's questions; every backend is tabulated in results.md.
# Categorical slots 1–6 of the lab palette, in their validated order.
SERIES = (
    ("cuda_flat", "#2a78d6", "o"),
    ("cuda_vec", "#eb6834", "D"),
    ("cuda_vec_ilp4", "#1baf7a", "s"),
    ("cuda_stream", "#eda100", "^"),
    ("triton_2d", "#e87ba4", "v"),
    ("cuda_2d_colmajor", "#008300", "X"),
)
REFERENCE = "copy_reference"
PANELS = (("fp32", "FP32"), ("fp16", "FP16 (BF16 is the same within noise)"))
STYLE = {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}
PEAK_GB_S = 960
L2_BYTES = 64 * 2**20


def load(root):
    values = {}
    gpu = None
    for path in sorted(Path(root).glob("device_*.json")):
        data = json.loads(path.read_text())
        gpu = data["environment"]["gpu"]
        for r in data["results"]:
            if r["status"] == "ok":
                key = (r["implementation"], r["dtype"], r["logical_bytes"])
                values.setdefault(key, []).append(r["effective_gb_s"])
    return {k: statistics.median(v) for k, v in values.items()}, gpu


def series(data, backend, dtype):
    points = sorted((b, gb) for (name, d, b), gb in data.items() if name == backend and d == dtype)
    return [b / 1e6 for b, _ in points], [gb for _, gb in points]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "artifacts/matrix_copy"
    data, gpu = load(root)
    t = STYLE
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.patch.set_facecolor(t["surface"])
    for ax, (dtype, title) in zip(axes, PANELS, strict=True):
        ax.set_facecolor(t["surface"])
        ax.axvspan(L2_BYTES / 1.5 / 1e6, L2_BYTES / 1e6, color=t["grid"], alpha=0.6, lw=0)
        ax.text(
            L2_BYTES / 1.22 / 1e6, 2780, "64 MiB L2\nfills", ha="center", va="bottom",
            fontsize=8, color=t["muted"],
        )  # fmt: skip
        ax.axhline(PEAK_GB_S, color=t["muted"], lw=1, ls=(0, (4, 3)))
        ax.text(1.1, PEAK_GB_S + 40, "960 GB/s DRAM peak", fontsize=8, color=t["muted"])
        x, y = series(data, REFERENCE, dtype)
        ax.plot(x, y, color=t["text"], lw=2, ls=(0, (1, 1.5)), label="memcpy (copy reference)")
        for backend, color, marker in SERIES:
            x, y = series(data, backend, dtype)
            ax.plot(
                x, y, color=color, lw=2, marker=marker, ms=6, mec=t["surface"], mew=1.2,
                label=backend,
            )  # fmt: skip
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontsize=10, color=t["text"])
        ax.set_xlabel("Logical traffic per call, 2·N²·e (MB, log scale)", color=t["muted"])
        ax.grid(True, which="major", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    axes[0].set_ylabel("Effective bandwidth, GB/s (higher is better)", color=t["muted"])
    axes[0].set_ylim(0, 3050)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, 0), handlelength=2.4,
    )  # fmt: skip
    for text in legend.get_texts():
        text.set_color(t["text"])
    fig.suptitle(
        f"Matrix copy on {gpu}: N = 512 to 8192, median of three runs, inputs reused, "
        "allocation included",
        fontsize=10,
        color=t["text"],
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    target = Path(__file__).parents[1] / "figures" / "bandwidth_vs_size.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, facecolor=t["surface"])
    fig.savefig(Path(root) / "bandwidth_vs_size.png", dpi=140, facecolor=t["surface"])


if __name__ == "__main__":
    main()
