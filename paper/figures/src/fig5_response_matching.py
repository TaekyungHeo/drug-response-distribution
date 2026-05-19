"""Generate Fig. 5: response profile matching K-curve on GDSC2.

Single panel — no panel label.
Output: paper/figures/fig5_response_matching.pdf + .png
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
DATA = ROOT / "experiments/05_solutions/03_few_shot/01_response_matching/report/data/results.json"

KS = [0, 1, 3, 5, 10, 20, 50]


def main() -> None:
    figure_style.apply()
    scale = 0.75
    for key in ("font.size", "axes.titlesize", "axes.labelsize",
                "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        matplotlib.rcParams[key] = matplotlib.rcParams[key] * scale

    with open(DATA) as f:
        res = json.load(f)

    ceiling = res["overall"]["measurement_ceiling"]
    curve = {entry["k"]: entry for entry in res["k_curve"]}

    ks = np.array(KS)
    rs = np.array([curve[k]["mean_r"] for k in KS])
    permuted_r = curve[50]["permuted_r"]

    fig, ax = plt.subplots(figsize=(figure_style.ONE_AND_HALF, 1.8))

    # Measurement ceiling
    ax.axhline(ceiling, color=figure_style.LGRAY, lw=0.8, ls=":", zorder=1,
               label=f"Ceiling ({ceiling:.3f})")

    # Cell-mean prior (K=0)
    prior = rs[0]
    ax.axhline(prior, color=figure_style.GRAY, lw=0.8, ls="--", zorder=1,
               label=f"Cell-mean prior ({prior:.3f})")

    # K-curve
    ax.plot(ks, rs, color=figure_style.BLUE, lw=figure_style.LINE,
            marker="o", markersize=figure_style.MARKER, zorder=3,
            label="Blended matching")

    # Permuted control at K=50
    ax.plot(50, permuted_r, marker="x", color=figure_style.RED,
            markersize=figure_style.MARKER + 1, markeredgewidth=0.9,
            zorder=4, label=f"Permuted $K=50$ ({permuted_r:.3f})")

    ax.set_xticks(KS)
    ax.set_xlabel("$K$ pilot observations")
    ax.set_ylabel("Per-drug $r$")
    ax.set_xlim(-2, 54)
    ax.set_ylim(0, 0.85)
    ax.legend(loc="lower right")
    figure_style.clean_axis(ax)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig5_response_matching.{ext}"
        figure_style.savefig(fig, out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
