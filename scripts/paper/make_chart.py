"""Regenerate paper/figures/core_metrics_vigalt.pdf from paired_statistical_tests.csv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "paired_statistical_tests.csv"
DEFAULT_PDF = ROOT / "paper" / "figures" / "core_metrics_vigalt.pdf"


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PDF

    rows = {r["metric"]: r for r in csv.DictReader(csv_path.open(encoding="utf-8"))}
    mets = ["Objectivity", "Relevancy", "Relevant Novelty", "Avoidable Redundancy"]
    labels = [
        "Objectivity",
        "Relevancy",
        "Relevant\nnovelty",
        "Avoidable\nredundancy\n(lower is better)",
    ]
    v = [float(rows[m]["vigalt_macro_mean"]) * 100 for m in mets]
    b = [float(rows[m]["caid_macro_mean"]) * 100 for m in mets]
    sig = [rows[m]["significant_after_holm_0.05"] == "True" for m in mets]

    plt.rcParams.update(
        {"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42}
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    x = np.arange(len(mets))
    w = 0.36
    gap = 0.02
    bv = ax.bar(x - w / 2 - gap / 2, v, w, color="#2a78d6", label="ViGALT", zorder=3)
    bb = ax.bar(x + w / 2 + gap / 2, b, w, color="#eb6834", label="CAID", zorder=3)
    for rect, val in list(zip(bv, v)) + list(zip(bb, b)):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            val + 1.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.3,
            color="#222",
        )
    for i, s in enumerate(sig):
        if s:
            top = max(v[i], b[i]) + 10
            ax.plot(
                [x[i] - w / 2, x[i] - w / 2, x[i] + w / 2, x[i] + w / 2],
                [top - 2, top, top, top - 2],
                color="#333",
                lw=0.6,
            )
            ax.text(x[i], top + 0.5, "*", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Macro-average score (%)", fontsize=7.5)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=7, length=2, width=0.5)
    ax.yaxis.grid(True, color="#dddddd", lw=0.5, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.08),
        handlelength=1.2,
        columnspacing=1,
    )
    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
