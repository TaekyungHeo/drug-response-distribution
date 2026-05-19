"""Generate Supplementary Fig. 1: top-correlated features per drug (target identity vs. lineage).

Produces:
  - paper/figures/sfig1_top_features.pdf
  - paper/figures/sfig1_top_features.png

Uses combined RNA + mutation features; Pearson |r| vs ln_IC50 per drug.
"""

from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figure_style
FIGURES = ROOT / "paper" / "figures"

DRUGS = ["Dabrafenib", "Afatinib", "Olaparib", "Trametinib", "Navitoclax"]
TARGETS = {
    "Dabrafenib": ("BRAF", "BRAF_mut"),
    "Afatinib": ("EGFR", "EGFR"),
    "Olaparib": ("BRCA", "BRCA2_mut"),
    "Trametinib": ("MEK", "MAP2K1_mut"),
    "Navitoclax": ("BCL2", "BCL2"),
}
DRUG_TITLE = {d: f"{d}\n(target: {TARGETS[d][0]})" for d in DRUGS}


def compute_top_features(drug: str, n: int = 10) -> tuple[list[str], list[float], int]:
    dr = pd.read_parquet(ROOT / "data" / "processed" / "drug_response.parquet")
    rna = pd.read_parquet(ROOT / "data" / "processed" / "rna.parquet")
    mut = pd.read_parquet(ROOT / "data" / "processed" / "mutations.parquet")
    mut_r = mut.add_suffix("_mut")

    dv = dr[dr["drug_name"] == drug].set_index("depmap_id")["ln_ic50"]
    common = dv.index.intersection(rna.index).intersection(mut_r.index)
    y = dv.loc[common].values
    X = pd.concat([rna.loc[common], mut_r.loc[common]], axis=1)

    cors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for feat in X.columns:
            x = X[feat].values
            if x.std() > 0:
                r, _ = stats.pearsonr(x, y)
                cors.append((feat, abs(r)))

    cors.sort(key=lambda t: t[1], reverse=True)
    target_key = TARGETS[drug][1]
    target_rank = next((i + 1 for i, (f, _) in enumerate(cors) if f == target_key), None)

    names = [f for f, _ in cors[:n]]
    values = [v for _, v in cors[:n]]
    return names, values, target_rank


def generate() -> None:
    figure_style.apply()
    fig, axes_grid = plt.subplots(2, 3, figsize=(figure_style.FULL, 5.25), sharey=False)
    axes = axes_grid.ravel()

    for ax, drug in zip(axes, DRUGS):
        names, values, target_rank = compute_top_features(drug)
        target_key = TARGETS[drug][1]

        colors = [figure_style.RED if n == target_key else figure_style.BLUE for n in names]

        # Rank 1 at top → reverse order for horizontal bar chart
        names_rev = names[::-1]
        values_rev = values[::-1]
        colors_rev = colors[::-1]
        # Display labels: replace "_mut" suffix with " (mut.)" for readability
        display_labels = [n.replace("_mut", " (mut.)") for n in names_rev]

        ax.barh(range(len(names_rev)), values_rev, color=colors_rev, alpha=0.9, edgecolor="none")
        ax.set_yticks(range(len(names_rev)))
        ax.set_yticklabels(display_labels, fontsize=5.4)
        ax.set_xlabel("|r|")
        ax.set_title(DRUG_TITLE[drug], fontweight="bold", pad=3)
        ax.set_xlim(0, max(values) * 1.2)
        figure_style.clean_axis(ax)

    axes[-1].axis("off")
    fig.subplots_adjust(hspace=0.42, wspace=0.38, bottom=0.11, top=0.92)

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=figure_style.RED, label="Known target"),
        Patch(facecolor=figure_style.BLUE, label="Other feature"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.005))

    for ext in ["pdf", "png"]:
        figure_style.savefig(fig, FIGURES / f"sfig1_top_features.{ext}")
    plt.close(fig)
    print("Saved sfig1_top_features.pdf/png")

    # Print target ranks for caption update
    print("\nTarget ranks:")
    for drug in DRUGS:
        _, _, rank = compute_top_features(drug)
        print(f"  {drug}: {TARGETS[drug][1]} rank = {rank}")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    generate()
    print("Done.")
