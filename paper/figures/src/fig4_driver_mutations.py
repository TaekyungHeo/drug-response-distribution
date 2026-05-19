"""Generate Fig. 4: oncogenic driver mutations stratify sensitivity to matched inhibitors.

Two panels:
  a — EGFR inhibitors: ln IC50 by EGFR mutation status (violin + box)
  b — ERK MAPK inhibitors: ln IC50 by BRAF/KRAS mutation status (violin + box)

Output: paper/figures/fig4_driver_mutations.pdf + .png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import figure_style

OUT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# Colorblind-friendly: orange (mutant) vs slate-blue (WT)
C_MUT = figure_style.ORANGE
C_WT = figure_style.SKY


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    return float((b.mean() - a.mean()) / np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2))


def pval_str(p: float) -> str:
    if p < 0.001:
        return f"p={p:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"p={p:.3f}"


def draw_violin(ax, pos: float, data: np.ndarray, color: str,
                width: float = 0.30, alpha: float = 0.85) -> None:
    parts = ax.violinplot([data], positions=[pos], widths=width,
                          showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        pc.set_alpha(alpha)
    # Box overlay
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    ax.plot([pos, pos], [q1, q3], color="k", lw=0.9, solid_capstyle="round", zorder=3)
    ax.plot(pos, med, "o", color="white", markeredgecolor="k",
            markeredgewidth=0.7, markersize=3.1, zorder=4)


def annotate_sig(ax, x1: float, x2: float, y: float,
                 p: float, d: float | None = None) -> None:
    h = 0.07 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ay = y + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [ay, ay + h, ay + h, ay],
            lw=0.6, color="k")
    label = pval_str(p)
    if d is not None:
        label += f"\nd = {d:.2f}"
    ax.text((x1 + x2) / 2, ay + h + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            label, ha="center", va="bottom", fontsize=5.5)


def load_data():
    dr = pd.read_parquet(DATA_DIR / "drug_response.parquet")
    mut = pd.read_parquet(DATA_DIR / "mutations.parquet")
    common = set(dr["depmap_id"].unique()) & set(mut.index)
    dr = dr[dr["depmap_id"].isin(common)].copy()
    return dr, mut, common


def panel_egfr(ax, dr: pd.DataFrame, mut: pd.DataFrame) -> None:
    drugs = ["Gefitinib", "Afatinib", "Erlotinib", "Osimertinib"]
    x_positions = np.arange(len(drugs))
    offset = 0.20

    for xi, drug_name in enumerate(drugs):
        sub = dr[dr["drug_name"] == drug_name]
        egfr_flag = mut.loc[sub["depmap_id"].values, "EGFR"].values
        wt_ic50  = sub["ln_ic50"].values[egfr_flag == 0]
        mut_ic50 = sub["ln_ic50"].values[egfr_flag == 1]

        draw_violin(ax, xi - offset, wt_ic50,  C_WT)
        draw_violin(ax, xi + offset, mut_ic50, C_MUT)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(drugs, rotation=35, ha="right")
    ax.set_ylabel("ln IC$_{50}$")
    ax.set_title("EGFR inhibitors", fontweight="bold", pad=10)
    ax.set_xlim(-0.6, len(drugs) - 0.4)
    ax.set_ylim(-6.4, 10.5)

    for xi, drug_name in enumerate(drugs):
        sub = dr[dr["drug_name"] == drug_name]
        egfr_flag = mut.loc[sub["depmap_id"].values, "EGFR"].values
        wt_ic50  = sub["ln_ic50"].values[egfr_flag == 0]
        mut_ic50 = sub["ln_ic50"].values[egfr_flag == 1]
        if len(mut_ic50) > 5:
            _, p = mannwhitneyu(mut_ic50, wt_ic50, alternative="less")
            d = cohen_d(mut_ic50, wt_ic50)
            ymax = max(wt_ic50.max(), mut_ic50.max())
            ax.annotate(f"{pval_str(p)}\nd={d:.2f}", xy=(xi, ymax + 0.1),
                        ha="center", va="bottom", fontsize=6.0, color="#333333")

    patches = [
        mpatches.Patch(color=C_WT,  label="WT"),
        mpatches.Patch(color=C_MUT, label="EGFR mut"),
    ]
    ax.legend(handles=patches, loc="lower left", frameon=False,
              handlelength=1, handleheight=0.8)
    figure_style.panel_label(ax, "a")
    figure_style.clean_axis(ax)


def panel_erk(ax, dr: pd.DataFrame, mut: pd.DataFrame) -> None:
    drugs = ["PD0325901", "Trametinib", "Selumetinib", "SCH772984"]
    x_positions = np.arange(len(drugs))
    offset = 0.20

    for xi, drug_name in enumerate(drugs):
        sub = dr[dr["drug_name"] == drug_name]
        if sub.empty:
            continue
        braf = mut.loc[sub["depmap_id"].values, "BRAF"].values
        kras = mut.loc[sub["depmap_id"].values, "KRAS"].values
        bk_flag = (braf == 1) | (kras == 1)
        wt_ic50  = sub["ln_ic50"].values[~bk_flag]
        mut_ic50 = sub["ln_ic50"].values[bk_flag]

        draw_violin(ax, xi - offset, wt_ic50,  C_WT)
        draw_violin(ax, xi + offset, mut_ic50, C_MUT)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(drugs, rotation=35, ha="right")
    ax.set_ylabel("ln IC$_{50}$")
    ax.set_title("ERK MAPK inhibitors", fontweight="bold", pad=10)
    ax.set_xlim(-0.6, len(drugs) - 0.4)
    ax.set_ylim(-9.2, 14.0)

    for xi, drug_name in enumerate(drugs):
        sub = dr[dr["drug_name"] == drug_name]
        if sub.empty:
            continue
        braf = mut.loc[sub["depmap_id"].values, "BRAF"].values
        kras = mut.loc[sub["depmap_id"].values, "KRAS"].values
        bk_flag = (braf == 1) | (kras == 1)
        wt_ic50  = sub["ln_ic50"].values[~bk_flag]
        mut_ic50 = sub["ln_ic50"].values[bk_flag]
        if len(mut_ic50) > 5 and len(wt_ic50) > 5:
            _, p = mannwhitneyu(mut_ic50, wt_ic50, alternative="less")
            d = cohen_d(mut_ic50, wt_ic50)
            ymax = max(wt_ic50.max(), mut_ic50.max())
            ax.annotate(f"{pval_str(p)}\nd={d:.2f}", xy=(xi, ymax + 0.15),
                        ha="center", va="bottom", fontsize=6.0, color="#333333")

    patches = [
        mpatches.Patch(color=C_WT,  label="WT"),
        mpatches.Patch(color=C_MUT, label="BRAF/KRAS mut"),
    ]
    ax.legend(handles=patches, loc="lower left", frameon=False,
              handlelength=1, handleheight=0.8)
    figure_style.panel_label(ax, "b")
    figure_style.clean_axis(ax)


def main() -> None:
    figure_style.apply()
    dr, mut, _ = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(figure_style.FULL, 2.3))
    panel_egfr(axes[0], dr, mut)
    panel_erk(axes[1], dr, mut)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig4_driver_mutations.{ext}"
        figure_style.savefig(fig, out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
