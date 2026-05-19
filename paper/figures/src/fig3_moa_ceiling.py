"""Generate Fig. 3: MoA as feature vs. training distribution.

Two panels:
  a — Bar chart: per-drug r and global r with vs. without MoA one-hot feature
  b — Grouped bar chart: 7 MoA classes under all-drug vs. within-MoA training,
      with pairwise profile concordance markers

Output: paper/figures/fig3_moa_ceiling.pdf + .png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import figure_style

OUT_DIR    = Path(__file__).resolve().parents[1]
ONEHOT     = ROOT / "experiments/05_solutions/02_training_distribution/03_onehot_control/report/data/results.json"
WITHIN_MOA = ROOT / "experiments/05_solutions/02_training_distribution/01_within_moa/report/data/results.json"
CEILING    = ROOT / "experiments/05_solutions/01_diagnosis/02_moa_ceiling/report/data/results.json"
PARTC      = ROOT / "experiments/03_drug_feature_null/03_model_robustness/report/data/partC_metrics.json"

# MoA classes shown in panel b (matching the published figure)
SHOW_MOAS = [
    "ERK MAPK signaling",
    "EGFR signaling",
    "PI3K/MTOR signaling",
    "Chromatin other",
    "Genome integrity",
    "Cell cycle",
    "Apoptosis regulation",
]
XLABELS = [
    "ERK MAPK",
    "EGFR",
    "PI3K/MTOR",
    "Chromatin\nother",
    "Genome\nintegrity",
    "Cell\ncycle",
    "Apoptosis",
]

FONT_SCALE = 1.15


def panel_a(ax, onehot: dict, partc: dict) -> None:
    """Three bar groups: Ridge per-drug, Ridge global, Transformer per-drug."""
    # Ridge values
    ridge_nodrug_pd = onehot["baseline"]["mean_per_drug_r"]
    ridge_onehot_pd = onehot["with_moa_onehot"]["mean_per_drug_r"]
    ridge_nodrug_gl = onehot["baseline"]["global_r"]
    ridge_onehot_gl = onehot["with_moa_onehot"]["global_r"]

    # Transformer per-drug values
    xfmr_real = partc["moa_onehot"]["mean"]
    xfmr_perm = partc["moa_permuted"]["mean"]
    xfmr_nodrug = xfmr_real - partc["moa_onehot"]["delta_vs_no_drug"]

    w = 0.22
    gap = 0.30   # gap between Ridge groups
    xgap = 0.50  # gap before Transformer group

    # x centres for pairs/triples
    x0 = 0.0                     # Ridge per-drug: no-drug, one-hot
    x1 = x0 + 2 * w + gap        # Ridge global: no-drug, one-hot
    x2 = x1 + 2 * w + xgap       # Transformer: no-drug, real, permuted

    # Ridge per-drug
    ax.bar(x0,       ridge_nodrug_pd, w, color=figure_style.GRAY)
    ax.bar(x0 + w,   ridge_onehot_pd, w, color=figure_style.ORANGE)
    # Ridge global
    ax.bar(x1,       ridge_nodrug_gl, w, color=figure_style.GRAY,   label="No MoA")
    ax.bar(x1 + w,   ridge_onehot_gl, w, color=figure_style.ORANGE, label="MoA feature (real)")
    # Transformer per-drug
    ax.bar(x2,       xfmr_nodrug,     w, color=figure_style.GRAY)
    ax.bar(x2 + w,   xfmr_real,       w, color=figure_style.ORANGE)
    ax.bar(x2 + 2*w, xfmr_perm,       w, color=figure_style.BLUE,   label="MoA permuted",
           hatch="//", edgecolor=figure_style.BLUE, linewidth=0.5)

    # Group x-tick centres
    ax.set_xticks([x0 + w/2, x1 + w/2, x2 + w])
    ax.set_xticklabels(["Ridge\nper-drug $r$", "Ridge\nglobal $r$",
                         "Transformer\nper-drug $r$"], fontsize=6.5 * FONT_SCALE)

    # Data labels above each bar
    all_bars = [
        (x0,       ridge_nodrug_pd), (x0 + w,   ridge_onehot_pd),
        (x1,       ridge_nodrug_gl), (x1 + w,   ridge_onehot_gl),
        (x2,       xfmr_nodrug),     (x2 + w,   xfmr_real),
        (x2 + 2*w, xfmr_perm),
    ]
    for bx, bv in all_bars:
        ax.text(bx, bv + 0.010, f"{bv:.2f}",
                ha="center", va="bottom", fontsize=5.5 * FONT_SCALE)

    # Bracket + P annotation — placed well above all labels
    y_ann = max(xfmr_real, xfmr_perm) + 0.060
    x_real_c = x2 + w + w / 2
    x_perm_c = x2 + 2 * w + w / 2
    ax.annotate("", xy=(x_perm_c, y_ann), xytext=(x_real_c, y_ann),
                arrowprops=dict(arrowstyle="-", lw=0.6, color=figure_style.BLACK))
    ax.text((x_real_c + x_perm_c) / 2, y_ann + 0.003, "$P = 0.014$ (one-sided)",
            ha="center", va="bottom", fontsize=6 * FONT_SCALE)

    # Vertical separator between Ridge and Transformer
    ax.axvline(x2 - xgap / 2, color=figure_style.LGRAY, lw=0.5, ls="--", zorder=0)

    ax.set_ylabel("Pearson $r$")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6 * FONT_SCALE, loc="upper left")
    figure_style.title(ax, "MoA as feature")
    figure_style.panel_label(ax, "a")
    figure_style.clean_axis(ax)


def panel_b(ax, moa_data: dict, ceil_data: dict) -> None:
    """Grouped bars: 7 MoA classes, all-drug vs within-MoA + concordance."""
    row_by_moa   = {r["moa"]: r for r in moa_data["per_moa"]}
    ceil_by_moa  = {r["moa"]: r["mean_r"] for r in ceil_data["per_moa"]}

    all_drug_rs = [row_by_moa[m]["all_drug_mean_r"]   for m in SHOW_MOAS]
    within_rs   = [row_by_moa[m]["within_moa_mean_r"] for m in SHOW_MOAS]
    concordances = [ceil_by_moa.get(m, None)          for m in SHOW_MOAS]
    all_drug_mean = moa_data["overall"]["all_drug_mean_r"]

    x = np.arange(len(SHOW_MOAS))
    w = 0.35
    ax.bar(x - w / 2, all_drug_rs, w, color=figure_style.GRAY,
           label="All-drug training")
    ax.bar(x + w / 2, within_rs, w, color=figure_style.BLUE,
           label="Within-MoA training")

    # Pairwise concordance as short red horizontal lines
    bar_half = w * 0.55
    for xi, conc in zip(x, concordances):
        if conc is not None:
            ax.plot([xi + w / 2 - bar_half, xi + w / 2 + bar_half],
                    [conc, conc], color=figure_style.RED, lw=1.2, zorder=4)

    # All-drug mean dashed reference
    ax.axhline(all_drug_mean, color=figure_style.LGRAY, lw=0.7, ls="--",
               zorder=0, label=f"All-drug mean ({all_drug_mean:.3f})")

    # Data labels above the taller of the two bars per group
    for xi, v_within, v_all in zip(x, within_rs, all_drug_rs):
        y_label = max(v_within, v_all) + 0.015
        ax.text(xi + w / 2, y_label, f"{v_within:.2f}",
                ha="center", va="bottom", fontsize=5.5 * FONT_SCALE)

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=6.5 * FONT_SCALE)
    ax.set_ylabel("Per-drug $r$")
    ax.set_ylim(0, 1.0)

    # Legend — include red line for pairwise ceiling
    red_line = mlines.Line2D([], [], color=figure_style.RED, lw=1.2,
                              label="Pairwise ceiling")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [red_line],
              labels=lbls + ["Pairwise ceiling"],
              fontsize=6 * FONT_SCALE, loc="upper right", ncol=2)

    figure_style.title(ax, "MoA as training distribution")
    figure_style.panel_label(ax, "b")
    figure_style.clean_axis(ax)


def main() -> None:
    figure_style.apply()
    for key in ("font.size", "axes.titlesize", "axes.labelsize",
                "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        matplotlib.rcParams[key] = matplotlib.rcParams[key] * FONT_SCALE

    with open(ONEHOT)     as f: onehot   = json.load(f)
    with open(WITHIN_MOA) as f: moa_data = json.load(f)
    with open(CEILING)    as f: ceil_data = json.load(f)
    with open(PARTC)      as f: partc    = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(figure_style.FULL, 2.4),
                             gridspec_kw={"width_ratios": [1.4, 2.8]})
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97,
                        top=0.92, bottom=0.15)

    panel_a(axes[0], onehot, partc)
    panel_b(axes[1], moa_data, ceil_data)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig3_moa_ceiling.{ext}"
        figure_style.savefig(fig, out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
