# Training Distribution

Source 1: MoA label as training data selector. Does restricting or reweighting training data by MoA improve per-drug r?

| # | Experiment | Question | Key result |
|---|-----------|---------|------------|
| 01 | `01_within_moa/` | Strict within-MoA LOO training | ERK +0.296, EGFR +0.375; overall +0.020 |
| 02 | `02_moa_weighted/` | Soft same-MoA upweighting (2x–20x) | Best per MoA (Mitosis 0.81) |
| 03 | `03_onehot_control/` | MoA one-hot as feature (representation control) | Δ=+0.000 per-drug r; representation ≠ distribution |
| — | *(removed)* | Transformer dissociation | Confirmed via `03_drug_feature_null/03_model_robustness` Part C (full 10-fold): MoA Δ=+0.015 |
