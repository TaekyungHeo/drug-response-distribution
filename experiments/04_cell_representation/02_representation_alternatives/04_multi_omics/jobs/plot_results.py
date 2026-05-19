"""Generate modality ablation plots for Phase 2-03.

Usage:
    python experiments/04_cell_representation/02_representation_alternatives/04_multi_omics/jobs/plot_results.py RESULTS_JSON
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ABLATION_ORDER = [
    "rna_only",
    "rna_mutations",
    "rna_cnv",
    "rna_metabolomics",
    "rna_rppa",
    "all_5_omics",
]

CONDITION_LABELS = {
    "rna_only":         "RNA",
    "rna_mutations":    "RNA\n+Mut",
    "rna_cnv":          "RNA\n+CNV",
    "rna_metabolomics": "RNA\n+Metab",
    "rna_rppa":         "RNA\n+RPPA",
    "all_5_omics":      "All 5",
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: plot_results.py RESULTS_JSON")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    results = json.loads(results_path.read_text())
    run_dir = results_path.parent

    splits = ["mixed_set", "cell_blind"]
    split_labels = ["Mixed-Set", "Cell-Blind"]
    models = ["omnicancer_v1", "drug_fp_mlp"]
    model_labels = ["TransformerEncoder", "DrugFP MLP (concat)"]
    model_colors = ["#1565c0", "#64b5f6"]
    model_styles = ["-o", "--s"]

    x_labels = [CONDITION_LABELS[c] for c in ABLATION_ORDER]
    x = np.arange(len(ABLATION_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, split, split_label in zip(axes, splits, split_labels):
        for model_key, label, color, style in zip(models, model_labels, model_colors, model_styles):
            rs = [
                results.get(cond, {}).get(split, {}).get(model_key, {}).get("pearson_r", float("nan"))
                for cond in ABLATION_ORDER
            ]
            ax.plot(x, rs, style, label=label, color=color, linewidth=2, markersize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel("Modality Subset", fontsize=10)
        ax.set_ylabel("Pearson r", fontsize=10)
        ax.set_title(f"{split_label}", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Modality Ablation: Does TransformerEncoder Scale Monotonically With More Omics?",
        fontsize=12,
    )
    fig.tight_layout()
    plot_path = run_dir / "modality_ablation_curves.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close(fig)

    # ── Grouped bar: RNA-only vs All-5 improvement per model and split ────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x2 = np.arange(len(splits))
    bar_w = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]
    combos = [
        ("TransformerEncoder RNA-only", "omnicancer_v1", "rna_only", "#aed6f1"),
        ("TransformerEncoder All-5",    "omnicancer_v1", "all_5_omics", "#1565c0"),
        ("MLP RNA-only",        "drug_fp_mlp",   "rna_only", "#fad7a0"),
        ("MLP All-5",           "drug_fp_mlp",   "all_5_omics", "#f57c00"),
    ]
    for (label, model_key, cond, color), offset in zip(combos, offsets):
        vals = [
            results.get(cond, {}).get(s, {}).get(model_key, {}).get("pearson_r", float("nan"))
            for s in splits
        ]
        bars = ax2.bar(x2 + offset * bar_w, vals, bar_w, label=label, color=color, alpha=0.9)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(["Mixed-Set", "Cell-Blind"], fontsize=11)
    ax2.set_ylabel("Pearson r", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("RNA-Only vs All-5-Omics: TransformerEncoder vs Concat MLP", fontsize=11)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    plot2_path = run_dir / "rna_vs_all5_comparison.png"
    fig2.savefig(plot2_path, dpi=150)
    print(f"Saved: {plot2_path}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
