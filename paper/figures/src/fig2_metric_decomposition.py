"""Generate Fig. 2: metric decomposition and drug/cell representation ablations.

Four panels:
  a — Bar chart: global r vs. per-drug r under standard and z-scored training
  b — Lollipop: delta per-drug r across drug representation types
  c — Horizontal bar: cell representation ablation (omics combinations)
  d — Horizontal bar: baselines and ceilings

Output: paper/figures/fig2_metric_decomposition.pdf + .png
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

DECOMP_DATA   = ROOT / "experiments/01_metric_decomposition/01_global_vs_perdrug/report/data/results.json"
REPR_DATA     = ROOT / "experiments/03_drug_feature_null/02_representation_ablation/report/data/metrics.json"
BASELINES     = ROOT / "experiments/01_metric_decomposition/03_baselines/report/data/metrics.json"
CEILING_DATA  = ROOT / "experiments/04_cell_representation/01_ceiling_characterization/04_measurement_noise/report/data/results.json"
CELLBLIND_DATA = ROOT / "experiments/04_cell_representation/01_ceiling_characterization/01_split_ceilings/report/data/results.json"
KSHOT_DATA    = ROOT / "experiments/05_solutions/03_few_shot/01_response_matching/report/data/results.json"

FONT_SCALE = 1.15


def panel_a(ax: plt.Axes, data: dict) -> None:
    """Paired bars: global r and per-drug r under standard vs. z-scored."""
    conditions = ["standard", "per_drug_zscore"]
    xlabels    = ["Standard", "Per-drug\nz-scored"]

    global_rs   = [data[c]["global_r_mean"] for c in conditions]
    global_sds  = [data[c]["global_r_std"]  for c in conditions]
    perdrug_rs  = [data[c]["per_drug_r_mean"] for c in conditions]
    perdrug_sds = [data[c]["per_drug_r_std"]  for c in conditions]

    x = np.arange(len(conditions))
    w = 0.35
    bars_g = ax.bar(x - w / 2, global_rs, w, yerr=global_sds,
                    color=figure_style.BLUE, error_kw=dict(lw=0.7, capsize=2),
                    label="Global $r$")
    bars_p = ax.bar(x + w / 2, perdrug_rs, w, yerr=perdrug_sds,
                    color=figure_style.ORANGE, error_kw=dict(lw=0.7, capsize=2),
                    label="Per-drug $r$")
    # Removed dashed reference line — the message is carried by bar heights alone.

    # Data labels above each bar (clear of error bar caps)
    for bar, v, sd in zip(list(bars_g) + list(bars_p),
                           global_rs + perdrug_rs,
                           global_sds + perdrug_sds):
        ax.text(bar.get_x() + bar.get_width() / 2, v + sd + 0.008,
                f"{v:.2f}", ha="center", va="bottom", fontsize=5 * FONT_SCALE)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Pearson $r$")
    ax.set_ylim(0, 0.78)
    ax.legend(fontsize=5.5 * FONT_SCALE, loc="upper left")
    figure_style.title(ax, "Metric decomposition")
    figure_style.panel_label(ax, "a")
    figure_style.clean_axis(ax)


def panel_b(ax: plt.Axes, data: dict) -> None:
    """Horizontal bars: absolute per-drug r for each drug representation type.

    Null result is visually immediate (all bars identical length).
    LINCS is annotated with the global r drop on the LINCS-covered subset,
    which rules out the mechanistic-content hypothesis.
    """
    conds  = ["morgan_fp", "chemberta", "chembl_targets", "lincs", "all_concat"]
    labels = ["Morgan FP", "ChemBERTa", "Drug targets", "LINCS", "All concat"]
    baseline = data["conditions"]["no_drug"]["mean"]
    values   = [data["conditions"][c]["mean"] for c in conds]

    XMAX = 0.80
    ys = np.arange(len(conds))
    ax.barh(ys, values, color=figure_style.BLUE, alpha=0.85)
    ax.axvline(baseline, color=figure_style.LGRAY, lw=0.7, ls="--", zorder=0)
    ax.set_xlim(0, XMAX)

    # Data labels at right end of each bar
    for y, v in zip(ys, values):
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=5 * FONT_SCALE)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Per-drug $r$")
    figure_style.title(ax, "Drug representations")
    figure_style.panel_label(ax, "b")
    figure_style.clean_axis(ax)


def panel_c(ax: plt.Axes, baselines: dict) -> None:
    """Horizontal bars: per-drug r by cell representation (omics ablation)."""
    abl = baselines["stage2_modality_ablation"]
    rna_r    = abl["rna_only"]["splits"]["drug_blind"]["per_drug_r"]
    rna_mut_r = abl["rna_mut"]["splits"]["drug_blind"]["per_drug_r"]
    cnv_r    = abl["rna_mut_cnv"]["splits"]["drug_blind"]["per_drug_r"]
    all_r    = abl["all_5_omics"]["splits"]["drug_blind"]["per_drug_r"]

    labels = ["RNA", "RNA+Mut", "RNA+Mut+CNV", "All 5 omics"]
    values = [rna_r, rna_mut_r, cnv_r, all_r]
    ref    = rna_mut_r

    ys = np.arange(len(labels))
    ax.barh(ys, values, color=figure_style.BLUE, alpha=0.85)
    ax.axvline(ref, color=figure_style.LGRAY, lw=0.7, ls="--", zorder=0)
    ax.set_xlim(0, 0.80)

    # Data labels at right end of each bar
    for y, v in zip(ys, values):
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=5 * FONT_SCALE)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Per-drug $r$")
    figure_style.title(ax, "Cell representation")
    figure_style.panel_label(ax, "c")
    figure_style.clean_axis(ax)


def panel_d(ax: plt.Axes, repr_data: dict, cellblind: dict,
            ceiling_data: dict, kshot_data: dict) -> None:
    """Horizontal bar: baselines and ceilings summary."""
    ceiling  = ceiling_data["per_drug_replicate_r"]["mean"]
    prior_r  = kshot_data["overall"]["cell_mean_prior_r"]
    ridge_r  = repr_data["conditions"]["no_drug"]["mean"]

    # cell-blind CV
    cb_folds = cellblind["cell_blind"]["folds"]
    cb_r     = float(np.mean([f["per_drug_r"] for f in cb_folds]))

    # Order top-to-bottom: highest first → reverse for barh (y=0 = bottom)
    labels = ["Cell-blind Ridge", "Cell-mean prior", "Ridge (no drug)"]
    values = [cb_r, prior_r, ridge_r]
    colors = [figure_style.GRAY, figure_style.GRAY, figure_style.GRAY]
    # Cell-blind Ridge: use hatching to distinguish without red
    hatches = ["//", "", ""]

    ys = np.arange(len(labels))
    for y, v, col, hatch in zip(ys, values, colors, hatches):
        ax.barh(y, v, color=col, hatch=hatch,
                edgecolor=figure_style.BLACK if hatch else "none",
                linewidth=0.3)
    ax.axvline(ceiling, color=figure_style.BLACK, lw=0.7, ls=":", zorder=3)
    ax.text(ceiling, len(labels) - 0.3, "ceiling",
            va="bottom", ha="center", fontsize=5.5 * FONT_SCALE, rotation=0, color=figure_style.BLACK)

    for y, v in zip(ys, values):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=5.5 * FONT_SCALE)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Per-drug $r$")
    ax.set_xlim(0, 0.80)
    figure_style.title(ax, "Baselines and ceilings")
    figure_style.panel_label(ax, "d")
    figure_style.clean_axis(ax)


def main() -> None:
    figure_style.apply()
    scale = 1.15
    for key in ("font.size", "axes.titlesize", "axes.labelsize",
                "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        matplotlib.rcParams[key] = matplotlib.rcParams[key] * scale

    with open(DECOMP_DATA)    as f: decomp       = json.load(f)
    with open(REPR_DATA)      as f: repr_data    = json.load(f)
    with open(BASELINES)      as f: baselines    = json.load(f)
    with open(CEILING_DATA)   as f: ceil_data    = json.load(f)
    with open(CELLBLIND_DATA) as f: cellblind    = json.load(f)
    with open(KSHOT_DATA)     as f: kshot        = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(figure_style.FULL, 4.6))
    fig.subplots_adjust(hspace=0.52, wspace=0.38,
                        left=0.08, right=0.97, top=0.93, bottom=0.1)

    panel_a(axes[0, 0], decomp)
    panel_b(axes[0, 1], repr_data)
    panel_c(axes[1, 0], baselines)
    panel_d(axes[1, 1], repr_data, cellblind, ceil_data, kshot)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig2_metric_decomposition.{ext}"
        figure_style.savefig(fig, out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
