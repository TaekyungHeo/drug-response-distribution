"""Generate Extended Data Fig. 1: PASO checkpoint selection inflation.

Two panels:
  a — Per-fold per-drug r: fair vs PASO-style (shows insensitivity)
  b — Global r inflation decomposition: fair mean → PASO-style mean → best fold

Output: paper/figures/edfig1_checkpoint_inflation.pdf + .png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import figure_style

OUT_DIR = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/02_reproductions/01_paso/02_decomposition/report/data/metrics.json"


def main() -> None:
    figure_style.apply()
    scale = 1.25
    for key in ("font.size", "axes.titlesize", "axes.labelsize",
                "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        matplotlib.rcParams[key] = matplotlib.rcParams[key] * scale

    with open(DATA) as f:
        m = json.load(f)

    folds = m["fold_details"]
    n = len(folds)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(figure_style.FULL, 2.4),
                                      gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(wspace=0.35, left=0.07, right=0.97, top=0.90, bottom=0.15)

    # Panel a: per-fold per-drug r
    x = np.arange(1, n + 1)
    fair_pdr = [f["fair_per_drug_r"] for f in folds]
    paso_pdr = [f["paso_style_per_drug_r"] for f in folds]

    w = 0.35
    ax_a.bar(x - w / 2, fair_pdr, w, color=figure_style.BLUE, label="Fair")
    ax_a.bar(x + w / 2, paso_pdr, w, color=figure_style.ORANGE, label="PASO-style")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(x)
    ax_a.set_xlabel("Fold")
    ax_a.set_ylabel("Per-drug $r$")
    ax_a.set_ylim(0, 0.85)
    ax_a.legend(fontsize=6 * scale)
    figure_style.title(ax_a, "Per-fold per-drug $r$")
    figure_style.panel_label(ax_a, "a")
    figure_style.clean_axis(ax_a)

    # Panel b: global r inflation decomposition
    fair_global = m["fair_mean"]
    paso_global = m["paso_style_mean"]
    best_fold = m["best_fold_test_r"]
    paso_reported = m["paso_reported_r"]

    labels = ["Fair\n(mean)", "PASO-style\n(mean)", "PASO\n(reported)"]
    values = [fair_global, paso_global, paso_reported]
    colors = [figure_style.BLUE, figure_style.ORANGE, figure_style.RED]

    bars = ax_b.bar(range(3), values, color=colors, width=0.6)
    for bar, v in zip(bars, values):
        ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                  f"{v:.3f}", ha="center", va="bottom", fontsize=6 * scale)

    ax_b.set_xticks(range(3))
    ax_b.set_xticklabels(labels, fontsize=6 * scale)
    ax_b.set_ylabel("Global $r$")
    ax_b.set_ylim(0, 0.85)
    figure_style.title(ax_b, "Inflation from leaky evaluation")
    figure_style.panel_label(ax_b, "b")
    figure_style.clean_axis(ax_b)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"edfig1_checkpoint_inflation.{ext}"
        figure_style.savefig(fig, out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
