"""Generate Fig. 6: response matching generalization across datasets.

Three panels:
  a — CTRPv2 (out-of-distribution cells)
  b — BeatAML (patient-derived samples)
  c — PRISM (scope failure)

Output: paper/figures/fig6_generalization.pdf + .png
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
GDSC2_DATA = ROOT / "experiments/05_solutions/03_few_shot/01_response_matching/report/data/results.json"
EXT_DATA   = ROOT / "experiments/06_external_validation/03_kshot_matching/report/data/results.json"

KS = [0, 1, 3, 5, 10, 20, 50]


def plot_kcurve(ax, ks, rs, prior, title_str, letter, ylim=None, label_fontsize=8) -> None:
    ax.axhline(prior, color=figure_style.GRAY, lw=0.8, ls="--", zorder=1)
    ax.plot(ks, rs, color=figure_style.BLUE, lw=figure_style.LINE,
            marker="o", markersize=figure_style.MARKER, zorder=3)
    ax.set_xticks([0, 5, 10, 20, 50])
    ax.set_xlabel("$K$ pilot observations")
    ax.set_xlim(-2, 54)
    if ylim:
        ax.set_ylim(*ylim)
    figure_style.title(ax, title_str)
    ax.text(-0.10, 1.04, letter, transform=ax.transAxes,
            fontsize=label_fontsize, fontweight="bold", va="top", ha="right",
            color=figure_style.BLACK, clip_on=False)
    figure_style.clean_axis(ax)


def main() -> None:
    figure_style.apply()
    scale = 0.8
    for key in ("font.size", "axes.titlesize", "axes.labelsize",
                "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        matplotlib.rcParams[key] = matplotlib.rcParams[key] * scale

    with open(EXT_DATA) as f:
        ext = json.load(f)

    ctrp  = {e["k"]: e["per_drug_r"] for e in ext["ctrpv2"]["k_curve"]}
    beat  = {e["k"]: e["per_drug_r"] for e in ext["beataml"]["k_curve"]}
    prism = {e["k"]: e["per_drug_r"] for e in ext["prism"]["k_curve"]}

    ks = np.array(KS)

    fig, axes = plt.subplots(1, 3, figsize=(figure_style.FULL * 0.75, 1.8))

    shared_ylim = (0.0, 0.56)

    label_fs = 8 * scale

    # a: CTRPv2
    ctrp_rs = np.array([ctrp[k] for k in KS])
    plot_kcurve(axes[0], ks, ctrp_rs, ctrp[0],
                "CTRPv2 (OOD cells)", "a", ylim=shared_ylim, label_fontsize=label_fs)
    axes[0].set_ylabel("Per-drug $r$")

    # b: BeatAML
    beat_rs = np.array([beat[k] for k in KS])
    plot_kcurve(axes[1], ks, beat_rs, beat[0],
                "BeatAML (patient-derived)", "b", ylim=shared_ylim, label_fontsize=label_fs)

    # c: PRISM
    prism_rs = np.array([prism[k] for k in KS])
    plot_kcurve(axes[2], ks, prism_rs, prism[0],
                "PRISM (scope failure)", "c", ylim=shared_ylim, label_fontsize=label_fs)

    # Legend inside panel c (large empty space above PRISM data)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=figure_style.BLUE, lw=figure_style.LINE,
               marker="o", markersize=figure_style.MARKER, label="Blended matching"),
        Line2D([0], [0], color=figure_style.GRAY, lw=0.8, ls="--",
               label="Cell-mean prior ($K=0$)"),
    ]
    axes[2].legend(handles=handles, loc="upper right", fontsize=6 * scale, frameon=False)

    for ext_str in ("pdf", "png"):
        out = OUT_DIR / f"fig6_generalization.{ext_str}"
        figure_style.savefig(fig, out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
