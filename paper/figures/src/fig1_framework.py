"""Generate Fig. 1: conceptual framework — global r vs. per-drug r schematic.

Single panel:
  a — Schematic: global r vs. per-drug r (potency + cell ranking vs. cell ranking only)

Output: paper/figures/fig1_framework.pdf + .png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import figure_style

OUT_DIR = Path(__file__).resolve().parents[1]


def panel_a_schematic(ax: plt.Axes) -> None:
    """Schematic: 3 drug strips at different heights (left) collapsed to same (right)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rng = np.random.default_rng(42)
    n_cells = 10
    colors = [figure_style.BLUE, figure_style.ORANGE, figure_style.GREEN]

    # Box boundaries (axes fraction)
    lx0, lx1 = 0.10, 0.46   # left box
    rx0, rx1 = 0.58, 0.98   # right box
    by0, by1 = 0.10, 0.86   # shared y bounds

    for bx0, bx1 in [(lx0, lx1), (rx0, rx1)]:
        rect = mpatches.FancyBboxPatch(
            (bx0, by0), bx1 - bx0, by1 - by0,
            boxstyle="round,pad=0.01", linewidth=0.5,
            edgecolor=figure_style.BLACK, facecolor="white",
            transform=ax.transAxes, zorder=2,
        )
        ax.add_patch(rect)

    # Drug mean y-positions: spread on left, collapsed on right
    drug_y_left  = [0.70, 0.48, 0.26]
    drug_y_right = [0.48, 0.48, 0.48]

    # Shared base cell scores — sorted so ordering is visible left-to-right
    base = np.linspace(-0.10, 0.10, n_cells)

    pad = 0.035
    cell_xs_l = np.linspace(lx0 + pad, lx1 - pad, n_cells)
    cell_xs_r = np.linspace(rx0 + pad, rx1 - pad, n_cells)

    for dy_l, dy_r, col in zip(drug_y_left, drug_y_right, colors):
        noise = rng.normal(0, 0.018, n_cells)
        ys_l = dy_l + base + noise
        ys_r = dy_r + base + noise  # same relative spread, different center

        ax.scatter(cell_xs_l, ys_l, s=13, color=col, alpha=0.9, zorder=3,
                   transform=ax.transAxes)
        ax.plot([lx0 + pad * 0.4, lx1 - pad * 0.4], [dy_l, dy_l],
                '--', color=col, lw=0.7, alpha=0.5, zorder=2,
                transform=ax.transAxes)

        ax.scatter(cell_xs_r, ys_r, s=13, color=col, alpha=0.9, zorder=3,
                   transform=ax.transAxes)

    # Shared center line on right
    ax.plot([rx0 + pad * 0.4, rx1 - pad * 0.4], [0.48, 0.48],
            '--', color=figure_style.LGRAY, lw=0.7, zorder=2,
            transform=ax.transAxes)

    # Potency double-arrow (left of left box)
    ax.annotate("", xy=(lx0 - 0.04, drug_y_left[0]),
                xytext=(lx0 - 0.04, drug_y_left[2]),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=figure_style.GRAY))
    ax.text(lx0 - 0.025, 0.48, "potency", rotation=90,
            ha="center", va="center", fontsize=6, color=figure_style.GRAY,
            transform=ax.transAxes)

    # Labels above boxes
    ax.text((lx0 + lx1) / 2, by1 + 0.03, "Global $r$",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            transform=ax.transAxes)
    ax.text((rx0 + rx1) / 2, by1 + 0.03, "Per-drug $r$",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            transform=ax.transAxes)

    # Descriptions below boxes
    ax.text((lx0 + lx1) / 2, by0 - 0.03, "Potency + cell ranking",
            ha="center", va="top", fontsize=6, color=figure_style.BLACK,
            transform=ax.transAxes)
    ax.text((rx0 + rx1) / 2, by0 - 0.03, "Cell ranking only",
            ha="center", va="top", fontsize=6, color=figure_style.BLACK,
            transform=ax.transAxes)

    # Arrow between boxes
    mid_y = 0.48
    ax.annotate("", xy=(rx0 - 0.01, mid_y), xytext=(lx1 + 0.01, mid_y),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=figure_style.BLACK))
    ax.text((lx1 + rx0) / 2, mid_y + 0.09, "center\neach drug",
            ha="center", va="bottom", fontsize=6, color=figure_style.BLACK,
            transform=ax.transAxes)


def main() -> None:
    figure_style.apply()

    fig, ax = plt.subplots(1, 1, figsize=(figure_style.FULL, 1.9))

    panel_a_schematic(ax)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig1_framework.{ext}"
        figure_style.savefig(fig, out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
