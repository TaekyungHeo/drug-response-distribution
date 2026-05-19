# Experiment Index

Experiments follow the paper's narrative: establish metric → audit prior work →
show drug features don't help → characterize cell ceiling → propose solutions → validate.

## Directory Map

```
experiments/
├── 00_data_preparation/          # Raw data download and preprocessing
│
├── 01_metric_decomposition/      # Establish per-drug r as primary metric
│   ├── 01_global_vs_perdrug/     # Between-drug variance dominates global r (68%)
│   ├── 02_metric_selection/      # Per-drug Pearson r selected over Spearman/NDCG/R²
│   └── 03_baselines/             # Cell-mean prior r=0.645 drug-blind CV (holdout oracle r=0.652)
│
├── 02_reproductions/             # Prior-work reproductions and bias audit
│   ├── 01_paso/                  # PASO snooping: fair r=0.509 vs reported 0.745
│   ├── 02_deepcdr/               # DeepCDR snooping audit
│   └── 03_drugcell/              # DrugCell snooping audit
│
├── 03_drug_feature_null/         # Drug representation ablations (null result)
│   ├── 01_oracle_bounds/         # Drug-mean oracle (global r=0.818), Tanimoto ceiling (0.718)
│   ├── 02_representation_ablation/  # 8 representations, all Δ=+0.001 (= degenerate baseline)
│   ├── 03_model_robustness/      # Transformer: Morgan Δ=+0.008 (null); LINCS Δ=-0.005, drug-target Δ=-0.019 (harmful); MoA Δ=+0.015 (marginal)
│   ├── 04_split_robustness/      # Scaffold-blind: Δ=+0.0006 (null holds)
│   ├── 05_dataset_robustness/    # PRISM within-dataset: Δ=+0.0165 at r=0.112 (contextual)
│   ├── 06_objective_axis/        # RankNet Δ=+0.014 vs no_drug: objective matters, not representation
│   └── 07_cross_dataset_transfer/  # GDSC2→PRISM, delta=+0.002
│
├── 04_cell_representation/       # Cell features: ceiling, alternatives, robustness
│   ├── 01_ceiling_characterization/  # drug-blind r=0.645, 86% of replicate ceiling (0.754)
│   ├── 02_representation_alternatives/  # PCA/scFoundation/RPPA/multi-omics all Δ≤+0.004
│   ├── 03_data_sufficiency/      # Learning curve: no plateau, data-limited component
│   └── 04_methodological_robustness/  # XGBoost/MLP/ranking-loss all Δ≈0
│
│
├── 05_solutions/                    # What moves per-drug r beyond cell-mean prior?
│   ├── 01_diagnosis/                # Landscape: which drugs are hard and why?
│   │   ├── 01_moa_performance/      # Per-MoA per-drug r under all-drug training
│   │   └── 02_moa_ceiling/          # Within-MoA profile concordance
│   ├── 02_training_distribution/    # Source 1: MoA → training data selection
│   │   ├── 01_within_moa/           # Strict within-MoA training
│   │   ├── 02_moa_weighted/         # Soft: upweight same-MoA pairs
│   │   └── 03_onehot_control/       # Control: MoA as feature ≠ distribution
│   ├── 03_few_shot/                 # Source 2: K observed responses
│   │   ├── 01_response_matching/    # K=0→50, blending, crossover
│   │   ├── 02_active_selection/     # Which cells to screen first?
│   │   └── 03_kshot_mechanism/      # K=1 patient-specific signal (BeatAML)
│   ├── 04_external_signatures/      # Source 3: functional drug profile
│   │   ├── 01_lincs/                # LINCS L1000 as drug feature: per-drug r Δ=+0.001 (null); global r Δ=-0.058
│   │   └── 02_lincs_prediction/     # Gate: predict LINCS from structure?
│   └── 05_combinations/             # Do solutions combine?
│       ├── 01_moa_x_kshot/          # Within-MoA + K-shot
│       └── 02_lincs_x_moa/          # LINCS + within-MoA
│
└── 06_external_validation/          # Cross-dataset replication (CTRPv2, BeatAML, PRISM)
    ├── 01_drug_feature_null/        # Drug feature null replication (all Δ≈0 or contextually irrelevant)
    ├── 02_moa_training/             # Within-MoA training replication (CTRPv2 EGFR Δ=+0.371 confirmed)
    └── 03_kshot_matching/           # K-shot K-curve: CTRPv2/BeatAML validated; PRISM scope failure (K≥3 collapse)
```

## Narrative Flow

```
01: "The field's metric is wrong"
 → 02: "The field's evaluations are wrong"
  → 03: "With the right metric, drug features don't help"
   → 04: "Cell representation hits cell-mean prior ceiling"
    → 05: "Training distribution and few-shot observation DO help"
     → 06: "Findings replicate on patients and other assays"
```

## Completion Status

| Group | Completed | In Progress | Queued |
|-------|:---------:|:-----------:|:------:|
| 01_metric_decomposition | 3/3 | — | — |
| 02_reproductions | 3/3 | — | — |
| 03_drug_feature_null | 7/7 | — | — |
| 04_cell_representation | 12/12 | — | — |
| 05_solutions | 12/12 | — | — |
| 06_external_validation | 3/3 | — | — |

## Notes

- 03_drug_feature_null: all 7 falsification axes addressed. 03_model_robustness Part C
  complete (Morgan Δ=+0.008, LINCS Δ=-0.005, drug-target Δ=-0.019, MoA Δ=+0.015 with
  permuted-MoA control confirming class identity used). GNN embeddings (Part B) below
  gate. 06 RankNet Δ=+0.014 — objective matters, not representation.
  05 PRISM Δ=+0.0165 at r=0.112 (see 05 README; paper's canonical PRISM values from
  06_external_validation: 0.117/0.134/+0.017).
- 04_cell_representation: all 12 leaf experiments complete. Learning curve (03/01) does not
  show a clear plateau — see `03_data_sufficiency/README.md`.
- 05_ceilings removed: measurement ceiling (replicate r=0.754) folded into 04/01/04.
- 06_active_learning folded into 05_solutions/03_few_shot/02_active_selection.
- 05_solutions: all 12 complete. 04_transformer_moa removed (incomplete smoke run); Transformer dissociation confirmed via 03_model_robustness Part C (full 10-fold, MoA Δ=+0.015).
