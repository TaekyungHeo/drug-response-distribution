# 05 Solutions — What moves per-drug r beyond the cell-mean prior?

## Context

01-04 established:
- Per-drug r is the correct metric (01)
- Cell-only ceiling = cell-mean prior = 0.645 (K=0 in PASO 10-fold drug-blind CV;
  03_baselines holdout oracle = 0.652 uses test-set means and is not a practical target)
- Drug features don't help per-drug r (03)
- Measurement ceiling = 0.754 (04)
- Gap: 0.754 - 0.645 = 0.109

To exceed 0.645, a model needs drug-specific information that changes cell rankings
beyond "this cell is generally sensitive/resistant." There are exactly three sources
of such information:

1. **Mechanism label** (MoA) — changes which drugs the model trains on
2. **Direct observation** (K responses) — the drug's actual IC50 on K cells
3. **Functional profile** (LINCS) — transcriptional effect of the drug on cells

## Structure (MECE)

```
05_solutions/
├── 01_diagnosis/                    # Landscape: which drugs are hard and why?
│   ├── 01_moa_performance/          # Per-MoA per-drug r under all-drug training
│   └── 02_moa_ceiling/              # Within-MoA profile concordance (biological ceiling)
│
├── 02_training_distribution/        # Source 1: MoA label → training data selection
│   ├── 01_within_moa/               # Strict: train only on same-MoA drugs
│   ├── 02_moa_weighted/             # Soft: upweight same-MoA pairs (2x–20x)
│   └── 03_onehot_control/           # Control: MoA label as feature, not distribution
│
├── 03_few_shot/                     # Source 2: direct observation → K-shot adaptation
│   ├── 01_response_matching/        # K=0→50, cell-mean blending, crossover analysis
│   ├── 02_active_selection/         # Which cells to screen for maximum K-shot gain?
│   └── 03_kshot_mechanism/          # K=1 in BeatAML: patient-specific vs potency signal
│
├── 04_external_signatures/          # Source 3: functional profile → drug feature
│   ├── 01_lincs/                    # LINCS null: per-drug r Δ=+0.001; global r Δ=-0.058
│   └── 02_lincs_prediction/         # Gate: can LINCS be predicted from structure?
│
└── 05_combinations/                 # Do solutions combine additively?
    ├── 01_moa_x_kshot/              # Within-MoA training + K-shot matching
    └── 02_lincs_x_moa/              # LINCS signatures + within-MoA training
```

## Key Results

| # | Experiment | Key result |
|---|-----------|------------|
| 01/01 | `moa_performance` | Overall mean r=0.645; top MoAs: Mitosis 0.77, Cell cycle 0.77 |
| 01/02 | `moa_ceiling` | Within-MoA concordance: Mitosis 0.71, EGFR 0.70 |
| 02/01 | `within_moa` | ERK MAPK: 0.427→0.723 (Δ=+0.296); EGFR: 0.425→0.799 (Δ=+0.375); overall mean 0.644→0.664 (Δ=+0.020) |
| 02/02 | `moa_weighted` | Soft MoA weighting: best per MoA (Mitosis 0.81) |
| 02/03 | `onehot_control` | MoA one-hot Δ=+0.000 per-drug r; +0.041 global r (representation ≠ distribution) |
| 02/04 | *(removed — smoke run)* | Transformer dissociation confirmed via `03_drug_feature_null/03_model_robustness` Part C (full 10-fold): MoA Δ=+0.015 |
| 03/01 | `response_matching` | K-curve: K=0→0.645, K=5→0.646, K=10→0.653, K=50→0.701; first exceeds cell-mean prior at K=5 (Δ=+0.001, marginal); reliable improvement from K=20 (Δ=+0.025) |
| 03/02 | `active_selection` | MaxVar at K=1: delta=+0.021 over random; no strategy consistently dominates across all K |
| 03/03 | `kshot_mechanism` | BeatAML K=1: cell-specific r=0.356 vs mean-potency r=0.298 (Δ=+0.059); patient-specific signal present |
| 04/01 | `lincs` | LINCS global r Δ=-0.058 (no_drug 0.301→lincs 0.243); per-drug r Δ=+0.001 (null); metric dissociation confirmed |
| 04/02 | `lincs_prediction` | LINCS from structure: R²=-0.03, cosine=0.15; gate passes (structure cannot predict LINCS — chemical and transcriptional spaces are decoupled) |
| 05/01 | `moa_x_kshot` | EGFR: within-MoA 0.799, within-MoA+K=20 combined 0.801 (matches within-MoA alone at lower data); ERK: within-MoA 0.723, K-shot K=50 0.813 (K-shot dominates, exceeds within-MoA) |
| 05/02 | `lincs_x_moa` | LINCS does not improve per-drug r; within-MoA improves per-drug r only; effects are on different axes (EGFR pdr: +0.271, ERK pdr: +0.357) |

## Key mechanism claim

The central finding is: **representation ≠ distribution.**

MoA one-hot as a feature (03_onehot_control): delta ≤ +0.002
MoA as training distribution selector (01_within_moa): large gains for targeted
kinase inhibitor classes (ERK MAPK, EGFR); no gain for Apoptosis regulation
(genuine biological limit) or cell-division classes already captured by pan-cancer
fragility signal.

Same information, opposite outcomes. The bottleneck is not WHAT the model knows
about the drug, but WHICH drugs the model trains on. Pan-cancer co-training suppresses
pathway-specific sensitivity signals; within-MoA training removes this suppression.

## Dependencies

All experiments use:
- GDSC2 drug response matrix (data/processed/drug_response.parquet)
- PASO 10-fold drug-blind splits (external/PASO/data/10_fold_data/drug_blind/)
- RNA PCA(550) + mutation PCA(200) cell features
- Per-drug Pearson r as primary metric
- Ridge(alpha=1.0) as default model unless stated otherwise

Shared code in src/:
- src/evaluation/per_drug.py — per_drug_r(), mean_per_drug_r()
- src/data/splits.py — fold loading
- src/data/omics_utils.py — load_omics, z_score_normalize
