"""Plot the K-sweep roofline and the N-sweep bandwidth curve from measured_summary.json.

python -m problems.convolution_1d.benchmarks.plot artifacts/convolution_1d

figures/roofline_vs_k.svg: N = 2^24, K = 1 to 2047. Top row: achieved TFLOP/s against K
under the roofline min(FMA peak, intensity · bandwidth), using the measured FFMA peak and
07's cuda_vec copy bandwidth at the same N. Bottom row: effective GB/s (logical bytes per
call over time) against the copy ceiling. Two panels per row, FP32 and FP16. One y-scale
per panel; the two measures never share an axis.

figures/bandwidth_vs_size.svg: K = 31, effective GB/s against logical traffic per call,
with the 64 MiB L2 marked, as in 08 · Matrix Transpose's figure.

Each point is the median of three run medians.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-7 of the lab palette, in their validated order (as in 08's figure).
SERIES = (
    ("cuda_naive", "#2a78d6", "o"),
    ("cuda_const", "#eb6834", "D"),
    ("cuda_smem", "#1baf7a", "s"),
    ("cuda_smem_reg", "#eda100", "^"),
    ("cuda_large_k", "#e87ba4", "v"),
    ("cuda_unrolled", "#008300", "X"),
    ("triton", "#6250d6", "P"),
)
PANELS = (("fp32", "FP32"), ("fp16", "FP16 (FP32 accumulation)"))
STYLE = {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}
L2_BYTES = 64 * 2**20


def style_axis(ax, t):
    ax.set_facecolor(t["surface"])
    ax.grid(True, which="major", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)


def points(rows, sweep, backend, dtype, x, y):
    data = sorted(
        (r[x], r[y])
        for r in rows
        if (r["sweep"], r["backend"], r["dtype"]) == (sweep, backend, dtype)
    )
    return [a for a, _ in data], [b for _, b in data]


def legend(fig, axes, t):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, 0), handlelength=2.4,
    )  # fmt: skip
    for text in leg.get_texts():
        text.set_color(t["text"])


def roofline(summary, root, t):
    rows = summary["benchmarks"]
    peak = summary["fma_peak"]["achieved_tflop_s"]
    n = next(r["n"] for r in rows if r["sweep"] == "ksweep")
    copy = {r["dtype"]: r for r in rows if r["backend"] == "copy_cuda_vec" and r["n"] == n}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    fig.patch.set_facecolor(t["surface"])
    for col, (dtype, title) in enumerate(PANELS):
        top, bottom = axes[0, col], axes[1, col]
        bw = copy[dtype]["effective_gb_s"]  # GB/s
        ks = sorted({r["k"] for r in rows if r["sweep"] == "ksweep" and r["dtype"] == dtype})
        dense = [ks[0] * (ks[-1] / ks[0]) ** (i / 200) for i in range(201)]
        e = 4 if dtype == "fp32" else 2
        intensity = [2 * k * (n - k + 1) / ((2 * n + 1) * e) for k in dense]
        roof = [min(peak, i * bw / 1e3) for i in intensity]
        ridge_k = peak * 1e3 / bw * e  # K where intensity · bw = peak (N >> K)
        for ax in (top, bottom):
            style_axis(ax, t)
            ax.axvline(ridge_k, color=t["muted"], lw=1, ls=(0, (2, 2)))
        top.plot(dense, roof, color=t["text"], lw=1.6, ls=(0, (5, 2)), label="roofline (measured)")
        top.text(
            ridge_k * 1.08, peak * 0.035, f"ridge K ≈ {ridge_k:.0f}", fontsize=8, color=t["muted"]
        )
        top.text(
            ks[0], peak * 1.12, f"FFMA peak {peak:.1f} TFLOP/s (measured)", fontsize=8,
            color=t["muted"],
        )  # fmt: skip
        bottom.axhline(bw, color=t["text"], lw=1.6, ls=(0, (5, 2)))
        bottom.text(
            ks[0], bw * 1.03, f"copy cuda_vec {bw:.0f} GB/s (ceiling, plain copy)", fontsize=8,
            color=t["muted"],
        )  # fmt: skip
        for backend, color, marker in SERIES:
            kw = dict(
                color=color, lw=2, marker=marker, ms=6, mec=t["surface"], mew=1.2, label=backend
            )
            top.plot(*points(rows, "ksweep", backend, dtype, "k", "tflop_s"), **kw)
            kw.pop("label")
            bottom.plot(*points(rows, "ksweep", backend, dtype, "k", "effective_gb_s"), **kw)
        top.set_xscale("log", base=2)
        top.set_yscale("log")
        top.set_ylim(peak / 400, peak * 2)
        top.set_title(title, loc="left", fontsize=10, color=t["text"])
        highest = max(
            r["effective_gb_s"] for r in rows if r["sweep"] == "ksweep" and r["dtype"] == dtype
        )
        bottom.set_ylim(0, max(bw, highest) * 1.12)
        bottom.set_xlabel("Filter length K (log scale)", color=t["muted"])
        bottom.set_xticks(ks)
        bottom.set_xticklabels([str(k) for k in ks])
    axes[0, 0].set_ylabel("Achieved TFLOP/s (log, higher is better)", color=t["muted"])
    axes[1, 0].set_ylabel("Effective GB/s, logical bytes / time", color=t["muted"])
    legend(fig, axes, t)
    fig.suptitle(
        f"1D convolution on {summary['environment'][0]['gpu']}, N = 2^24: compute and bandwidth "
        "against K (median of three runs)",
        fontsize=10, color=t["text"], x=0.01, ha="left",
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, root, "roofline_vs_k", t)


def bandwidth(summary, root, t):
    rows = summary["benchmarks"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.patch.set_facecolor(t["surface"])
    top = max(r["effective_gb_s"] for r in rows if r["sweep"] in ("nsweep", "copy")) * 1.1
    for ax, (dtype, title) in zip(axes, PANELS, strict=True):
        style_axis(ax, t)
        ax.axvspan(L2_BYTES / 1.5 / 1e6, L2_BYTES / 1e6, color=t["grid"], alpha=0.6, lw=0)
        ax.text(
            L2_BYTES / 1.22 / 1e6, top * 0.9, "64 MiB L2\nfills", ha="center", va="bottom",
            fontsize=8, color=t["muted"],
        )  # fmt: skip
        x, y = points(rows, "copy", "copy_cuda_vec", dtype, "logical_bytes", "effective_gb_s")
        ax.plot(
            [v / 1e6 for v in x], y, color=t["text"], lw=1.6, ls=(0, (5, 2)),
            label="ceiling: copy cuda_vec (07, plain copy)",
        )  # fmt: skip
        for backend, color, marker in SERIES:
            x, y = points(rows, "nsweep", backend, dtype, "logical_bytes", "effective_gb_s")
            if not x:
                continue
            ax.plot(
                [v / 1e6 for v in x], y, color=color, lw=2, marker=marker, ms=6, mec=t["surface"],
                mew=1.2, label=backend,
            )  # fmt: skip
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontsize=10, color=t["text"])
        ax.set_xlabel("Logical traffic per call, (2N + 1)·e (MB, log scale)", color=t["muted"])
    axes[0].set_ylabel("Effective GB/s at K = 31 (higher is better)", color=t["muted"])
    axes[0].set_ylim(0, top)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, 0), handlelength=2.4,
    )  # fmt: skip
    for text in leg.get_texts():
        text.set_color(t["text"])
    fig.suptitle(
        f"1D convolution on {summary['environment'][0]['gpu']}, K = 31, N = 256 Ki to 64 Mi: "
        "inputs reused, allocation included, median of three runs",
        fontsize=10, color=t["text"], x=0.01, ha="left",
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, root, "bandwidth_vs_size", t)


def save(fig, root, name, t):
    target = Path(__file__).parents[1] / "figures" / f"{name}.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, facecolor=t["surface"])
    fig.savefig(root / f"{name}.png", dpi=140, facecolor=t["surface"])


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/convolution_1d")
    root.mkdir(parents=True, exist_ok=True)
    summary = json.loads((Path(__file__).parent / "measured_summary.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    roofline(summary, root, STYLE)
    bandwidth(summary, root, STYLE)


if __name__ == "__main__":
    main()
