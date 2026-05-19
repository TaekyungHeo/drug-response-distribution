"""Generate Supplementary Fig. 2: per-drug Pearson r by GDSC2 pathway annotation.

Each dot = one drug (drug-blind evaluation, cell-only baseline).
Dots coloured by performance tier:
  blue   : r > 0.65
  yellow : 0.50 – 0.65
  orange : r < 0.50
Pathways sorted descending by median r.
Dashed vertical line at the overall mean.

Produces:
  - paper/figures/sfig2_perdrug_by_pathway.pdf
  - paper/figures/sfig2_perdrug_by_pathway.png
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figure_style
FIGURES = ROOT / "paper" / "figures"

PER_DRUG_R_PATH = (
    ROOT / "experiments" / "03_drug_feature_null"
    / "02_representation_ablation" / "report" / "data" / "pooled_per_drug_r.json"
)
GDSC2_PATH = ROOT / "data" / "raw" / "GDSC2_fitted_dose_response_24Jul22.csv"

# Tier thresholds and colours
T_HIGH = 0.65
T_MID = 0.50
C_HIGH = figure_style.BLUE
C_MID = "#E6B800"   # warm yellow (distinguishable on white background)
C_LOW = figure_style.ORANGE

# Pathways with very few drugs are grouped into a readable label
MIN_N = 2


def _tier_color(r: float) -> str:
    if r > T_HIGH:
        return C_HIGH
    if r >= T_MID:
        return C_MID
    return C_LOW


def load_data() -> dict[str, list[tuple[str, float]]]:
    """Return {pathway: [(drug_name, per_drug_r), ...]} sorted by descending median."""
    with open(PER_DRUG_R_PATH) as f:
        per_drug_r: dict[str, float] = json.load(f)["no_drug"]

    gdsc = pd.read_csv(GDSC2_PATH, usecols=["DRUG_NAME", "PATHWAY_NAME"])
    pathway_map: dict[str, str] = (
        gdsc.drop_duplicates("DRUG_NAME")
        .set_index("DRUG_NAME")["PATHWAY_NAME"]
        .to_dict()
    )

    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for drug, r in per_drug_r.items():
        pathway = pathway_map.get(drug)
        if pathway is None:
            continue
        grouped[pathway].append((drug, r))

    # Drop pathways with only one drug (too sparse to show)
    grouped = {p: v for p, v in grouped.items() if len(v) >= MIN_N}

    # Sort pathways by descending median r
    sorted_pathways = sorted(
        grouped.keys(),
        key=lambda p: np.median([r for _, r in grouped[p]]),
        reverse=True,
    )
    return {p: grouped[p] for p in sorted_pathways}


def make_figure(grouped: dict[str, list[tuple[str, float]]]) -> matplotlib.figure.Figure:
    figure_style.apply()
    scale = 2.0
    for key in ("font.size", "axes.titlesize", "axes.labelsize",
                "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        matplotlib.rcParams[key] = matplotlib.rcParams[key] * scale

    n_pathways = len(grouped)
    fig_height = max(3.5, 0.32 * n_pathways)
    fig, ax = plt.subplots(figsize=(figure_style.FULL, fig_height))

    all_r = [r for drugs in grouped.values() for _, r in drugs]
    overall_mean = float(np.mean(all_r))

    pathway_names = list(grouped.keys())

    for yi, pathway in enumerate(pathway_names):
        drugs = grouped[pathway]
        rs = np.array([r for _, r in drugs])
        n = len(rs)

        # Jitter x positions slightly to avoid overlap
        rng = np.random.default_rng(seed=42 + yi)
        jitter = rng.uniform(-0.15, 0.15, size=n) if n > 1 else np.zeros(1)

        for r, j in zip(rs, jitter):
            ax.scatter(
                r, yi + j,
                color=_tier_color(r),
                s=9,
                linewidths=0,
                zorder=3,
                alpha=0.85,
            )

        # Median marker
        med = float(np.median(rs))
        ax.plot(med, yi, "|", color="k",
                markersize=5, markeredgewidth=0.8, zorder=4)

    # Overall mean dashed line
    ax.axvline(overall_mean, linestyle="--", linewidth=0.65,
               color=figure_style.GRAY, zorder=1)
    ax.text(
        overall_mean + 0.005, -0.7,
        f"mean = {overall_mean:.3f}",
        fontsize=5 * scale, color=figure_style.GRAY, va="top",
    )

    # Axes
    ax.set_yticks(range(n_pathways))
    ax.set_yticklabels(pathway_names, fontsize=5.5 * scale)
    ax.set_xlabel("Drug-blind per-drug Pearson r")
    ax.set_xlim(0.0, 0.95)
    ax.set_ylim(-1.0, n_pathways - 0.2)
    ax.invert_yaxis()

    # Legend
    handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_HIGH,
                      markersize=4, label=r"$r > 0.65$"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_MID,
                      markersize=4, label=r"$0.50 \leq r \leq 0.65$"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_LOW,
                      markersize=4, label=r"$r < 0.50$"),
        mlines.Line2D([], [], marker="|", color="k", markersize=5,
                      markeredgewidth=0.8, label="Median"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=5.5 * scale, handlelength=1.0)

    figure_style.clean_axis(ax)
    ax.tick_params(axis="y", length=0)  # remove y-tick marks for cleaner look

    return fig


def main() -> None:
    grouped = load_data()
    fig = make_figure(grouped)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIGURES / f"sfig2_perdrug_by_pathway.{ext}"
        figure_style.savefig(fig, out)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
